#!/usr/bin/env python3
"""
Whisper-PTT (CUDA): push-to-talk voice-to-text using faster-whisper on CUDA.
Hold hotkey → speak → release → transcription pasted into the active window.

Config: WHISPER_PTT_* env vars or .env file (see .env.example-cuda).

Dependencies: faster_whisper, pyaudio, keyboard, pyperclip, requests.
Optional: Ollama for LLM cleanup.
"""

import io
import os
import wave
import time
import threading
import collections
import keyboard
import pyaudio
import pyperclip
import requests
from faster_whisper import WhisperModel

# Load .env from script directory (so it works regardless of CWD)
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
        return False  # any other value → off
    if type_ is int:
        return int(raw)
    if type_ is float:
        return float(raw)
    return str(raw)


# -----------------------------------------------------------------------------
# Config (from env; values below are defaults)
# -----------------------------------------------------------------------------

# Whisper (CUDA only — no CPU fallback)
WHISPER_MODEL = _env("WHISPER_MODEL", "large-v3")
WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE", "en")
WHISPER_INITIAL_PROMPT = _env("WHISPER_INITIAL_PROMPT", "English speech.")

# Hotkey (hold to record, release to stop). Default: alt
HOTKEY = _env("HOTKEY", "alt").strip().lower().replace(" ", "")
# Parse combo (e.g. "ctrl+f12" -> ("ctrl", "f12")) for hook; single key -> (None, hotkey)
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
# Clipboard after paste (only when Paste is on): restore | clear | preserve
CLIPBOARD_AFTER_PASTE_POLICY = _env("CLIPBOARD_AFTER_PASTE_POLICY", "restore").strip().lower()
if CLIPBOARD_AFTER_PASTE_POLICY not in ("restore", "clear", "preserve"):
    raise SystemExit(
        f"Invalid config: CLIPBOARD_AFTER_PASTE_POLICY must be one of restore, clear, preserve (got {CLIPBOARD_AFTER_PASTE_POLICY!r})."
    )
# Keys after paste: key(s) to send (e.g. enter, ctrl+enter). Empty or "none" = no key.
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


# -----------------------------------------------------------------------------
# Windows: add CUDA DLL path (nvidia.* packages)
# -----------------------------------------------------------------------------

def _setup_cuda_dll_path():
    """Add nvidia.cublas/cudnn/cuda_runtime bin dirs to PATH for DLL loading."""
    for name in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"):
        try:
            mod = __import__(name, fromlist=[""])
            bin_dir = os.path.join(mod.__path__[0], "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
        except ImportError:
            pass


_setup_cuda_dll_path()


# -----------------------------------------------------------------------------
# Recording state and prebuffer
# -----------------------------------------------------------------------------

_recording = False
_audio_frames = []
_prebuffer_deque = None
_prebuffer_lock = threading.Lock()
_prebuffer_running = True
_pyaudio_instance = None
_whisper_model = None


def _prebuffer_size():
    return max(1, int(PREBUFFER_SEC * SAMPLE_RATE / CHUNK_SIZE))


# -----------------------------------------------------------------------------
# Audio: prebuffer and WAV
# -----------------------------------------------------------------------------

def _open_microphone_stream():
    return _pyaudio_instance.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )


def prebuffer_worker():
    """Background thread: read mic into ring buffer; when recording, also append to _audio_frames."""
    global _recording, _audio_frames
    stream = _open_microphone_stream()
    while _prebuffer_running:
        try:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
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


def frames_to_wav(frames, prepend_silence_sec=0):
    """Bytes frames list → WAV in memory (BytesIO). Optionally prepend silence."""
    if prepend_silence_sec > 0:
        sample_width = _pyaudio_instance.get_sample_size(AUDIO_FORMAT)
        silence_len = int(prepend_silence_sec * SAMPLE_RATE) * sample_width
        frames = [b"\x00" * silence_len] + list(frames)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(_pyaudio_instance.get_sample_size(AUDIO_FORMAT))
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"".join(frames))
    buf.seek(0)
    return buf


# -----------------------------------------------------------------------------
# Transcription and LLM
# -----------------------------------------------------------------------------

def transcribe(wav_buffer):
    """Transcribe WAV with Whisper. Returns (text, language_code)."""
    print("🔄 Transcribing...")
    t0 = time.time()
    segments, info = _whisper_model.transcribe(
        wav_buffer,
        language=WHISPER_LANGUAGE,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"📝 Whisper ({time.time() - t0:.1f}s): {text}")
    return text, info.language


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

def paste_to_front(text):
    """Copy to clipboard and/or paste to active window (Ctrl+V). If KEYS_AFTER_PASTE set, send that key(s) after paste."""
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
        keyboard.send("ctrl+v")
        time.sleep(0.1)
        if KEYS_AFTER_PASTE:
            time.sleep(0.05)
            keyboard.send(KEYS_AFTER_PASTE)
        suffix = f' + "{KEYS_AFTER_PASTE.upper()}"' if KEYS_AFTER_PASTE else ""
        print(f"✅ Pasted to active window{suffix}!")
        if CLIPBOARD_AFTER_PASTE_POLICY == "restore":
            pyperclip.copy(old)
        elif CLIPBOARD_AFTER_PASTE_POLICY == "clear":
            pyperclip.copy("")


# -----------------------------------------------------------------------------
# Process recording (background thread)
# -----------------------------------------------------------------------------

def _process_recorded_frames(frames):
    """Pipeline: frames → WAV → Whisper → optional LLM → paste."""
    wav = frames_to_wav(frames, prepend_silence_sec=PADDING_SEC)
    raw_text, lang = transcribe(wav)
    if USE_LLM_CLEANUP and raw_text.strip():
        final_text = cleanup_with_llm(raw_text, lang)
    else:
        final_text = raw_text
    paste_to_front(final_text)


def stop_recording_and_process():
    """Stop recording, wait for last frames, then transcribe and paste in background."""
    global _recording
    if not _recording:
        return
    _recording = False
    time.sleep(0.15)

    frames = list(_audio_frames)
    duration_sec = len(frames) * CHUNK_SIZE / SAMPLE_RATE
    print(f"⏹️ Recorded {duration_sec:.1f}s (with {PREBUFFER_SEC}s prebuffer)")

    if len(frames) < MIN_FRAMES:
        print("❌ Recording too short")
        return

    threading.Thread(target=_process_recorded_frames, args=(frames,), daemon=True).start()


# -----------------------------------------------------------------------------
# Hotkey and banner
# -----------------------------------------------------------------------------

def _on_hotkey_press(_event=None):
    if not _recording:
        if HOTKEY_MODIFIER is None or keyboard.is_pressed(HOTKEY_MODIFIER):
            start_recording()


def _on_hotkey_release(_event=None):
    stop_recording_and_process()


def _format_banner():
    w = 70
    def line(s, width=None):
        width = width or w
        padded = (s + " " * width)[:width]
        return "║" + padded + "║"
    parts = [
        "╔" + "═" * w + "╗\n",
        line("     🎤 Whisper-PTT ready!", w - 1) + "\n",
        line("") + "\n",
        line(f'     Hotkey: "{HOTKEY.upper()}" (hold to record, release to transcribe)') + "\n",
        line(f"     LLM cleanup: {'ON' if USE_LLM_CLEANUP else 'OFF'}") + "\n",
        line(f"     Copy to clipboard: {'ON' if COPY_TO_CLIPBOARD else 'OFF'}") + "\n",
        line(f"     Paste to active window: {'ON' if PASTE_TO_ACTIVE_WINDOW else 'OFF'}") + "\n",
    ]
    if PASTE_TO_ACTIVE_WINDOW:
        parts.append((line(f'     Keys after paste: "{KEYS_AFTER_PASTE.upper()}"') if KEYS_AFTER_PASTE else line("     Keys after paste: —")) + "\n")
    parts.extend([line("") + "\n", line('     "CTRL+C" to exit') + "\n", "╚" + "═" * w + "╝"])
    return "".join(parts)


def main():
    global _pyaudio_instance, _whisper_model, _prebuffer_deque

    print("⏳ Loading Whisper model... (first run may download the model)")
    _whisper_model = WhisperModel(
        WHISPER_MODEL,
        device="cuda",
        compute_type="float16",
    )
    print("✅ Whisper loaded!")

    _pyaudio_instance = pyaudio.PyAudio()
    _prebuffer_deque = collections.deque(maxlen=_prebuffer_size())

    print(f"🎧 Prebuffer active (last {PREBUFFER_SEC}s)")
    threading.Thread(target=prebuffer_worker, daemon=True).start()

    print(_format_banner())
    print(f'👂 Listening — hold "{HOTKEY.upper()}" to start recording.')

    # When hotkey is Pause, suppress it so the terminal doesn't freeze (Pause normally pauses terminal output)
    _suppress_pause = (HOTKEY_KEY == "pause")
    keyboard.on_press_key(HOTKEY_KEY, _on_hotkey_press, suppress=_suppress_pause)
    keyboard.on_release_key(HOTKEY_KEY, _on_hotkey_release, suppress=_suppress_pause)

    keyboard.wait("esc")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        raise SystemExit(0)
