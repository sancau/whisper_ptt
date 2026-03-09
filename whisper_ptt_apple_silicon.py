#!/usr/bin/env python3
"""
Whisper-PTT (Apple Silicon): push-to-talk voice-to-text using mlx-whisper on Metal.
Hold hotkey -> speak -> release -> transcription pasted into the active window.

Config: WHISPER_PTT_* env vars or .env file (see .env.example-apple-silicon).

Dependencies: mlx-whisper, pyaudio, keyboard, pyperclip, requests.
Optional: Ollama for LLM cleanup.
"""

import os
import queue
import subprocess
import time
import threading
import collections
import numpy as np
import pyaudio
import pyperclip
import requests
import mlx_whisper

_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_script_dir, ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(_env_path)
except ImportError:
    if os.path.isfile(_env_path):
        print()
        print("  ❌  NOTE: You have a .env file but python-dotenv is not installed, so it is not loaded.")
        print("      Config comes from environment variables only.")
        print("      To use .env, run:  pip install python-dotenv")
        print()


def _env(key, default, *, type_=str):
    """Read env var with type coercion. WHISPER_PTT_ prefix is optional."""
    full_key = key if key.startswith("WHISPER_PTT_") else f"WHISPER_PTT_{key}"
    raw = os.environ.get(full_key, os.environ.get(key, default))
    if type_ is bool:
        s = str(raw).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
        return False
    if type_ is int:
        return int(raw)
    if type_ is float:
        return float(raw)
    return str(raw)


# -----------------------------------------------------------------------------
# Config (from env; values below are defaults)
# -----------------------------------------------------------------------------

# Whisper (DEVICE and COMPUTE_TYPE not needed — MLX uses Metal automatically)
WHISPER_MODEL = _env("WHISPER_MODEL", "large-v3-turbo")
WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE", "en")
WHISPER_INITIAL_PROMPT = _env("WHISPER_INITIAL_PROMPT", "English speech.")

# Hotkey (hold to record, release to stop). Default: option
HOTKEY = _env("HOTKEY", "option").strip().lower().replace(" ", "")
if "+" in HOTKEY:
    _parts = HOTKEY.split("+", 1)
    HOTKEY_MODIFIER, HOTKEY_KEY = _parts[0].strip(), _parts[1].strip()
else:
    HOTKEY_MODIFIER, HOTKEY_KEY = None, HOTKEY

# LLM cleanup (Ollama) — optional, OFF by default
USE_LLM_CLEANUP = _env("USE_LLM_CLEANUP", "false", type_=bool)
OLLAMA_MODEL = _env("OLLAMA_MODEL", "gemma3:12b")
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_LLM_CLEANUP_PROMPT = """Fix the following speech-to-text transcription. Rules:
- Fix grammar, punctuation, and capitalization
- Remove filler words (um, uh, like, etc.)
- Keep the original language ({detected_lang})
- Keep the original meaning — do NOT add or change content
- If it's already clean, return as-is
- Return ONLY the cleaned text, nothing else

Transcription: {raw_text}"""
LLM_CLEANUP_PROMPT = _env("LLM_CLEANUP_PROMPT", DEFAULT_LLM_CLEANUP_PROMPT)

# Output: copy to clipboard and/or paste to active window
COPY_TO_CLIPBOARD = _env("COPY_TO_CLIPBOARD", "true", type_=bool)
PASTE_TO_ACTIVE_WINDOW = _env("PASTE_TO_ACTIVE_WINDOW", "true", type_=bool)
CLIPBOARD_AFTER_PASTE_POLICY = _env("CLIPBOARD_AFTER_PASTE_POLICY", "restore").strip().lower()
if CLIPBOARD_AFTER_PASTE_POLICY not in ("restore", "clear", "preserve"):
    raise SystemExit(
        f"Invalid config: CLIPBOARD_AFTER_PASTE_POLICY must be one of restore, clear, preserve (got {CLIPBOARD_AFTER_PASTE_POLICY!r})."
    )
KEYS_AFTER_PASTE = _env("KEYS_AFTER_PASTE", "enter").strip().lower()
if KEYS_AFTER_PASTE in ("", "none"):
    KEYS_AFTER_PASTE = None

# Audio
SAMPLE_RATE = _env("SAMPLE_RATE", "16000", type_=int)
CHANNELS = 1
CHUNK_SIZE = _env("CHUNK_SIZE", "1024", type_=int)
AUDIO_FORMAT = pyaudio.paInt16

# Prebuffer and padding
PREBUFFER_SEC = _env("PREBUFFER_SEC", "0.5", type_=float)
PADDING_SEC = _env("PADDING_SEC", "0.2", type_=float)
MIN_FRAMES = _env("MIN_FRAMES", "5", type_=int)
# Simple silence gate: max int16 amplitude below this is treated as silence.
SILENCE_AMPLITUDE_THRESHOLD = _env("SILENCE_AMPLITUDE", "750", type_=int)


# -----------------------------------------------------------------------------
# MLX model name → HuggingFace repo mapping
# -----------------------------------------------------------------------------

_MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en",
    "base": "mlx-community/whisper-base",
    "base.en": "mlx-community/whisper-base.en",
    "small": "mlx-community/whisper-small",
    "small.en": "mlx-community/whisper-small.en",
    "medium": "mlx-community/whisper-medium",
    "medium.en": "mlx-community/whisper-medium.en",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-turbo",
}


def _resolve_model(name):
    """Resolve short model name to mlx-community HuggingFace repo. Pass-through if already a repo path."""
    return _MLX_MODEL_MAP.get(name, name)


# -----------------------------------------------------------------------------
# Recording state and prebuffer
# -----------------------------------------------------------------------------

_recording = False
_audio_frames = []
_prebuffer_deque = None
_prebuffer_lock = threading.Lock()
_prebuffer_running = True
_pyaudio_instance = None
_mlx_model_path = None
_transcribe_queue = queue.Queue()
_model_ready = threading.Event()


def _prebuffer_size():
    return max(1, int(PREBUFFER_SEC * SAMPLE_RATE / CHUNK_SIZE))


# -----------------------------------------------------------------------------
# Audio: prebuffer and numpy conversion
# -----------------------------------------------------------------------------

_MIC_MAX_RETRIES = 5
_MIC_RETRY_DELAY = 1.0


def _open_microphone_stream():
    """Open mic stream, retrying with PyAudio re-init on transient PortAudio errors."""
    global _pyaudio_instance
    for attempt in range(1, _MIC_MAX_RETRIES + 1):
        try:
            return _pyaudio_instance.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )
        except OSError as e:
            print(f"⚠️  Mic open failed (attempt {attempt}/{_MIC_MAX_RETRIES}): {e}")
            try:
                _pyaudio_instance.terminate()
            except Exception:
                pass
            time.sleep(_MIC_RETRY_DELAY * attempt)
            _pyaudio_instance = pyaudio.PyAudio()
    raise RuntimeError("Could not open microphone after retries — check audio permissions and device.")


def prebuffer_worker():
    """Background thread: read mic into ring buffer; when recording, also append to _audio_frames."""
    global _recording, _audio_frames
    stream = _open_microphone_stream()
    while _prebuffer_running:
        try:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        except OSError as e:
            print(f"⚠️  Mic read error: {e} — reopening stream...")
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            try:
                stream = _open_microphone_stream()
            except RuntimeError:
                print("❌ Mic recovery failed, prebuffer stopping.")
                return
            continue
        except Exception:
            break
        with _prebuffer_lock:
            _prebuffer_deque.append(chunk)
            if _recording:
                _audio_frames.append(chunk)
    stream.stop_stream()
    stream.close()


def start_recording():
    """Start recording: copy prebuffer into _audio_frames; _recording flag lets worker append."""
    global _recording, _audio_frames
    with _prebuffer_lock:
        _audio_frames[:] = list(_prebuffer_deque)
    _recording = True
    print("🎙️ Recording...")


def frames_to_numpy(frames, prepend_silence_sec=0):
    """Raw PCM int16 frames → float32 numpy array normalised to [-1, 1]."""
    raw = b"".join(frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if prepend_silence_sec > 0:
        silence = np.zeros(int(prepend_silence_sec * SAMPLE_RATE), dtype=np.float32)
        audio = np.concatenate([silence, audio])
    return audio


# -----------------------------------------------------------------------------
# Transcription and LLM
# -----------------------------------------------------------------------------

def transcribe(audio_np):
    """Transcribe float32 numpy audio with mlx-whisper. Returns (text, language_code)."""
    print("🔄 Transcribing...")
    t0 = time.time()
    result = mlx_whisper.transcribe(
        audio_np,
        path_or_hf_repo=_mlx_model_path,
        language=WHISPER_LANGUAGE,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        fp16=True,
    )
    text = result["text"].strip()
    lang = result.get("language", WHISPER_LANGUAGE)
    print(f"📝 Whisper ({time.time() - t0:.1f}s): {text}")
    return text, lang


def cleanup_with_llm(raw_text, detected_lang):
    """LLM cleanup: fix punctuation, capitalization, remove filler words (Ollama)."""
    if not raw_text.strip():
        return raw_text
    print("🧹 LLM cleanup...")
    t0 = time.time()
    prompt = LLM_CLEANUP_PROMPT.format(detected_lang=detected_lang, raw_text=raw_text)
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": len(raw_text) * 2},
            },
            timeout=30,
        )
        result = r.json()["response"].strip()
        print(f"✨ LLM ({time.time() - t0:.1f}s): {result}")
        return result
    except Exception as e:
        print(f"❌ LLM error: {e}, using raw text")
        return raw_text


# -----------------------------------------------------------------------------
# Output: clipboard and/or paste to active window
# -----------------------------------------------------------------------------

def _applescript_keystroke(key, modifier="command"):
    """Send a keystroke via AppleScript (works reliably from any thread on macOS)."""
    modifiers = {
        "command": "command down",
        "control": "control down",
        "option": "option down",
        "shift": "shift down",
    }
    using = modifiers.get(modifier)
    if using:
        script = f'tell application "System Events" to keystroke "{key}" using {{{using}}}'
    else:
        script = f'tell application "System Events" to keystroke "{key}"'
    subprocess.run(["osascript", "-e", script], check=False)


def _applescript_key_code(key_name):
    """Send a special key (return, tab, etc.) via AppleScript key code."""
    key_codes = {
        "enter": 36, "return": 36,
        "tab": 48,
        "escape": 53,
        "space": 49,
        "delete": 51,
    }
    code = key_codes.get(key_name)
    if code is not None:
        script = f'tell application "System Events" to key code {code}'
        subprocess.run(["osascript", "-e", script], check=False)
    else:
        subprocess.run(
            ["osascript", "-e", f'tell application "System Events" to keystroke "{key_name}"'],
            check=False,
        )


def _send_keys_after_paste():
    """Parse KEYS_AFTER_PASTE (e.g. 'enter', 'ctrl+enter') and send via AppleScript."""
    if not KEYS_AFTER_PASTE:
        return
    parts = KEYS_AFTER_PASTE.split("+")
    if len(parts) == 1:
        _applescript_key_code(parts[0])
    else:
        modifier = parts[0].replace("ctrl", "control").replace("cmd", "command")
        key = parts[1]
        key_codes = {"enter": 36, "return": 36, "tab": 48}
        code = key_codes.get(key)
        modifiers_map = {
            "command": "command down",
            "control": "control down",
            "option": "option down",
            "shift": "shift down",
        }
        using = modifiers_map.get(modifier, "command down")
        if code is not None:
            script = f'tell application "System Events" to key code {code} using {{{using}}}'
        else:
            script = f'tell application "System Events" to keystroke "{key}" using {{{using}}}'
        subprocess.run(["osascript", "-e", script], check=False)


def paste_to_front(text):
    """Copy to clipboard and/or paste to active window (Cmd+V via AppleScript on macOS)."""
    if not text.strip():
        print("❌ Empty text, skipping")
        return
    if not COPY_TO_CLIPBOARD and not PASTE_TO_ACTIVE_WINDOW:
        print("✅ Done (console only)")
        return
    old = pyperclip.paste()
    pyperclip.copy(text)
    if COPY_TO_CLIPBOARD:
        print("📋 Copied to clipboard!")
    if PASTE_TO_ACTIVE_WINDOW:
        _applescript_keystroke("v", "command")
        time.sleep(0.1)
        if KEYS_AFTER_PASTE:
            time.sleep(0.05)
            _send_keys_after_paste()
        suffix = f' + "{KEYS_AFTER_PASTE.upper()}"' if KEYS_AFTER_PASTE else ""
        print(f"✅ Pasted to active window{suffix}!")
        if CLIPBOARD_AFTER_PASTE_POLICY == "restore":
            pyperclip.copy(old)
        elif CLIPBOARD_AFTER_PASTE_POLICY == "clear":
            pyperclip.copy("")


# -----------------------------------------------------------------------------
# Transcription worker (all MLX/Metal ops on a single thread)
# -----------------------------------------------------------------------------

def _transcription_worker():
    """Persistent thread owning all MLX operations — Metal requires same-thread access."""
    global _mlx_model_path
    _mlx_model_path = _resolve_model(WHISPER_MODEL)
    print(f"⏳ Loading mlx-whisper model '{_mlx_model_path}'... (first run downloads from HuggingFace)")
    warmup_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    mlx_whisper.transcribe(
        warmup_audio,
        path_or_hf_repo=_mlx_model_path,
        language=WHISPER_LANGUAGE,
        fp16=True,
        verbose=False,
    )
    print("✅ mlx-whisper loaded!")
    _model_ready.set()

    while True:
        frames = _transcribe_queue.get()
        if frames is None:
            break
        audio_np = frames_to_numpy(frames, prepend_silence_sec=PADDING_SEC)
        raw_text, lang = transcribe(audio_np)
        if USE_LLM_CLEANUP and raw_text.strip():
            final_text = cleanup_with_llm(raw_text, lang)
        else:
            final_text = raw_text
        paste_to_front(final_text)


def stop_recording_and_process():
    """Stop recording, wait for last frames, then enqueue for transcription."""
    global _recording
    if not _recording:
        return
    _recording = False
    time.sleep(0.15)

    frames = list(_audio_frames)
    duration_sec = len(frames) * CHUNK_SIZE / SAMPLE_RATE
    print(f"⏹️ Recorded {duration_sec:.1f}s (with {PREBUFFER_SEC}s prebuffer)")

    # Only process recordings longer than 0.7 seconds in total.
    if duration_sec <= 0.7 or len(frames) < MIN_FRAMES:
        print("❌ Recording too short")
        return

    # Simple silence / noise gate: skip very low-energy audio.
    raw = b"".join(frames)
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    if audio_int16.size == 0 or np.max(np.abs(audio_int16)) < SILENCE_AMPLITUDE_THRESHOLD:
        print("❌ Audio too quiet / silence, skipping")
        return

    _transcribe_queue.put(frames)


# -----------------------------------------------------------------------------
# Hotkey and banner
# -----------------------------------------------------------------------------

def _on_hotkey_press(_event=None):
    if not _recording:
        start_recording()


def _on_hotkey_release(_event=None):
    stop_recording_and_process()


def _start_hotkey_listener_mac():
    """Hotkey listener using pynput on macOS (no root required, Option supported)."""
    try:
        from pynput import keyboard as pynput_keyboard
    except ImportError:
        print("❌ pynput is required on macOS. Install with:")
        print("   pip install pynput")
        return

    Key = pynput_keyboard.Key
    pressed = set()

    def _spec_from_name(name):
        if not name:
            return None
        n = str(name).strip().lower()
        if n in ("cmd", "command", "⌘"):
            return Key.cmd
        if n in ("option", "opt", "alt", "⌥"):
            return Key.alt
        if n in ("ctrl", "control"):
            return Key.ctrl
        if n == "shift":
            return Key.shift
        if n.startswith("f") and n[1:].isdigit():
            return getattr(Key, n, None)
        if len(n) == 1:
            return n
        return None

    hotkey_key_spec = _spec_from_name(HOTKEY_KEY)
    hotkey_mod_spec = _spec_from_name(HOTKEY_MODIFIER) if HOTKEY_MODIFIER else None

    def _matches(key, spec):
        if spec is None:
            return False
        if isinstance(spec, str):
            return getattr(key, "char", None) == spec
        return key == spec

    def on_press(key):
        pressed.add(key)
        if HOTKEY_MODIFIER is None:
            if _matches(key, hotkey_key_spec):
                _on_hotkey_press()
        else:
            if _matches(key, hotkey_key_spec) and any(
                _matches(k, hotkey_mod_spec) for k in pressed
            ):
                _on_hotkey_press()

    def on_release(key):
        if key == Key.esc:
            print("\n👋 Exiting...")
            return False
        if HOTKEY_MODIFIER is None:
            if _matches(key, hotkey_key_spec):
                _on_hotkey_release()
        else:
            if _matches(key, hotkey_key_spec) or _matches(key, hotkey_mod_spec):
                _on_hotkey_release()
        pressed.discard(key)

    with pynput_keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def _format_banner():
    w = 70
    def line(s, width=None):
        width = width or w
        padded = (s + " " * width)[:width]
        return "║" + padded + "║"
    parts = [
        "╔" + "═" * w + "╗\n",
        line("     🎤 Whisper-PTT (Apple Silicon / MLX) ready!", w - 1) + "\n",
        line("") + "\n",
        line(f'     Hotkey: "{HOTKEY.upper()}" (hold to record, release to transcribe)') + "\n",
        line(f"     Model: {_mlx_model_path}") + "\n",
        line(f"     LLM cleanup: {'ON' if USE_LLM_CLEANUP else 'OFF'}") + "\n",
        line(f"     Copy to clipboard: {'ON' if COPY_TO_CLIPBOARD else 'OFF'}") + "\n",
        line(f"     Paste to active window: {'ON' if PASTE_TO_ACTIVE_WINDOW else 'OFF'}") + "\n",
    ]
    if PASTE_TO_ACTIVE_WINDOW:
        parts.append((line(f'     Keys after paste: "{KEYS_AFTER_PASTE.upper()}"') if KEYS_AFTER_PASTE else line("     Keys after paste: —")) + "\n")
    parts.extend([line("") + "\n", line('     "CTRL+C" to exit') + "\n", "╚" + "═" * w + "╝"])
    return "".join(parts)


def main():
    global _pyaudio_instance, _prebuffer_deque

    threading.Thread(target=_transcription_worker, daemon=True).start()
    _model_ready.wait()

    _pyaudio_instance = pyaudio.PyAudio()
    _prebuffer_deque = collections.deque(maxlen=_prebuffer_size())

    print(f"🎧 Prebuffer active (last {PREBUFFER_SEC}s)")
    threading.Thread(target=prebuffer_worker, daemon=True).start()

    print(_format_banner())
    print(f'👂 Listening — hold "{HOTKEY.upper()}" to start recording.')

    _start_hotkey_listener_mac()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        raise SystemExit(0)
