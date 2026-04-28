#!/usr/bin/env python3
"""
Whisper-PTT (CUDA): push-to-talk voice-to-text using faster-whisper on CUDA.
Hold hotkey → speak → release → transcription pasted into the active window.

Config: WHISPER_PTT_* env vars or .env file (see .env.example-cuda).

Dependencies: faster_whisper, pyaudio, keyboard, pyperclip, requests.
Optional: Ollama for LLM transform.
"""

import io
import os
import shutil
import subprocess
import wave
import time
import threading
import collections
from typing import overload
import keyboard
import pyaudio
import pyperclip
import requests
import numpy as np
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


@overload
def _env(key: str, default: str, *, type_: type[str] = str) -> str: ...
@overload
def _env(key: str, default: str, *, type_: type[bool]) -> bool: ...
@overload
def _env(key: str, default: str, *, type_: type[int]) -> int: ...
@overload
def _env(key: str, default: str, *, type_: type[float]) -> float: ...


def _env(key: str, default: str, *, type_: type = str):
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

# LLM transform (Ollama) — optional, OFF by default
USE_LLM_TRANSFORM = _env("USE_LLM_TRANSFORM", "false", type_=bool)
OLLAMA_MODEL = _env("OLLAMA_MODEL", "gemma3:12b")
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_LLM_TRANSFORM_PROMPT = """Fix the following speech-to-text transcription. Rules:
- Fix grammar, punctuation, and capitalization
- Remove filler words (um, uh, like, etc.)
- Keep the original language ({detected_lang})
- Keep the original meaning — do NOT add or change content
- If it's already clean, return as-is
- Return ONLY the cleaned text, nothing else

Transcription: {raw_text}"""
LLM_TRANSFORM_PROMPT = _env("LLM_TRANSFORM_PROMPT", DEFAULT_LLM_TRANSFORM_PROMPT)

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

# Paste shortcut (sent via `keyboard.send` after clipboard is set).
# Terminals (Konsole, xterm, gnome-terminal, etc.) need Ctrl+Shift+V because
# plain Ctrl+V in readline is `quoted-insert` and produces garbage characters.
DEFAULT_PASTE_SHORTCUT = _env("PASTE_SHORTCUT", "ctrl+v").strip().lower()
TERMINAL_PASTE_SHORTCUT = _env("TERMINAL_PASTE_SHORTCUT", "ctrl+shift+v").strip().lower()
AUTO_DETECT_TERMINAL = _env("AUTO_DETECT_TERMINAL", "true", type_=bool)

# WM_CLASS values reported by common X11 terminal emulators (lowercase).
TERMINAL_WM_CLASSES = {
    "konsole", "yakuake", "xterm", "uxterm", "urxvt", "rxvt",
    "gnome-terminal", "gnome-terminal-server", "terminator",
    "tilix", "xfce4-terminal", "alacritty", "kitty", "wezterm",
    "st-256color", "st", "qterminal", "lxterminal", "kgx",
}

# Audio
MIC_SAMPLE_RATE = 48000  # Hardware sample rate (USB mic supports 44100/48000)
SAMPLE_RATE = _env("SAMPLE_RATE", "16000", type_=int)  # Whisper target rate
CHANNELS = 1
CHUNK_SIZE = _env("CHUNK_SIZE", "1024", type_=int)
AUDIO_FORMAT = pyaudio.paInt16
INPUT_DEVICE_INDEX = _env("INPUT_DEVICE_INDEX", "").strip()
INPUT_DEVICE_NAME = _env("INPUT_DEVICE_NAME", "").strip().lower()
AUTO_SELECT_INPUT_DEVICE = _env("AUTO_SELECT_INPUT_DEVICE", "true", type_=bool)
LIST_AUDIO_DEVICES = _env("LIST_AUDIO_DEVICES", "false", type_=bool)

# Prebuffer and padding
PREBUFFER_SEC = _env("PREBUFFER_SEC", "0.5", type_=float)
PADDING_SEC = _env("PADDING_SEC", "0.2", type_=float)
MIN_FRAMES = _env("MIN_FRAMES", "5", type_=int)
# Simple silence gate: max int16 amplitude below this is treated as silence.
SILENCE_AMPLITUDE_THRESHOLD = _env("SILENCE_AMPLITUDE", "750", type_=int)


# -----------------------------------------------------------------------------
# Windows: add CUDA DLL path (nvidia.* packages)
# -----------------------------------------------------------------------------

def _setup_cuda_dll_path():
    """Add nvidia.cublas/cudnn/cuda_runtime bin dirs to PATH for DLL loading."""
    cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ]
    for path in cuda_paths:
        if os.path.isdir(path):
            try:
                os.add_dll_directory(path)
            except Exception:
                pass

    cuda_lib = os.environ.get("LD_LIBRARY_PATH", "")
    for path in cuda_paths:
        if path not in cuda_lib:
            cuda_lib += os.pathsep + path
    os.environ["LD_LIBRARY_PATH"] = cuda_lib

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
_prebuffer_deque: collections.deque[bytes] | None = None
_prebuffer_lock = threading.Lock()
_prebuffer_running = True
_pyaudio_instance: pyaudio.PyAudio | None = None
_whisper_model: WhisperModel | None = None


def _prebuffer_size():
    return max(1, int(PREBUFFER_SEC * SAMPLE_RATE / CHUNK_SIZE))


# -----------------------------------------------------------------------------
# Audio: prebuffer and WAV
# -----------------------------------------------------------------------------

def _input_devices():
    """Return PyAudio input devices visible in the current runtime."""
    devices = []
    if _pyaudio_instance is None:
        return devices
    for index in range(_pyaudio_instance.get_device_count()):
        try:
            info = _pyaudio_instance.get_device_info_by_index(index)
        except Exception:
            continue
        if int(info.get("maxInputChannels", 0)) <= 0:
            continue
        devices.append(info)
    return devices


def _print_input_devices():
    devices = _input_devices()
    if not devices:
        print("❌ No input audio devices found.")
        return
    print("🎚️ Input audio devices visible to this process:")
    for info in devices:
        index = int(info["index"])
        name = info["name"]
        channels = int(info.get("maxInputChannels", 0))
        rate = int(float(info.get("defaultSampleRate", 0)))
        print(f"  [{index}] {name} (inputs={channels}, default_rate={rate})")


def _auto_select_input_device_index():
    if not AUTO_SELECT_INPUT_DEVICE:
        return None
    preferred_tokens = ("usb pnp", "usb", "microphone", "mic", "capture")
    ignored_tokens = ("pulse", "default", "sysdefault")
    for token in preferred_tokens:
        for info in _input_devices():
            name = str(info.get("name", "")).lower()
            if token not in name:
                continue
            if any(ignored in name for ignored in ignored_tokens):
                continue
            return int(info["index"])
    return None


def _resolve_input_device_index():
    if INPUT_DEVICE_INDEX:
        try:
            return int(INPUT_DEVICE_INDEX)
        except ValueError:
            raise SystemExit(f"Invalid config: INPUT_DEVICE_INDEX must be an integer (got {INPUT_DEVICE_INDEX!r}).")
    if not INPUT_DEVICE_NAME:
        return _auto_select_input_device_index()
    for info in _input_devices():
        if INPUT_DEVICE_NAME in str(info.get("name", "")).lower():
            return int(info["index"])
    _print_input_devices()
    raise SystemExit(f"Input device matching {INPUT_DEVICE_NAME!r} was not found.")


def _open_microphone_stream():
    assert _pyaudio_instance is not None
    input_device_index = _resolve_input_device_index()
    if input_device_index is None:
        print("🎙️ Input device: system default")
    else:
        info = _pyaudio_instance.get_device_info_by_index(input_device_index)
        print(f"🎙️ Input device: [{input_device_index}] {info['name']}")
    return _pyaudio_instance.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=MIC_SAMPLE_RATE,
        input=True,
        input_device_index=input_device_index,
        frames_per_buffer=CHUNK_SIZE,
    )


def prebuffer_worker():
    """Background thread: read mic into ring buffer; when recording, also append to _audio_frames."""
    global _recording, _audio_frames
    assert _prebuffer_deque is not None
    prebuffer_deque = _prebuffer_deque
    stream = _open_microphone_stream()
    while _prebuffer_running:
        try:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        except Exception:
            break
        with _prebuffer_lock:
            prebuffer_deque.append(chunk)
            if _recording:
                _audio_frames.append(chunk)
    stream.stop_stream()
    stream.close()


def _resample_audio(audio_int16, src_rate, dst_rate):
    """Simple linear resampling from src_rate to dst_rate."""
    if src_rate == dst_rate:
        return audio_int16
    ratio = src_rate / dst_rate
    n_dst = int(len(audio_int16) / ratio)
    if n_dst == 0:
        return audio_int16
    indices = np.arange(n_dst) * ratio
    indices_floor = indices.astype(np.int32)
    indices_ceil = np.minimum(indices_floor + 1, len(audio_int16) - 1)
    frac = indices - indices_floor
    audio_float = audio_int16.astype(np.float32)
    resampled = audio_float[indices_floor] * (1 - frac) + audio_float[indices_ceil] * frac
    return resampled.astype(np.int16)


def start_recording():
    """Start recording: copy prebuffer into _audio_frames; _recording flag lets worker append."""
    global _recording, _audio_frames
    assert _prebuffer_deque is not None
    with _prebuffer_lock:
        _audio_frames[:] = list(_prebuffer_deque)
    _recording = True
    print("🎙️ Recording...")


def frames_to_wav(frames, prepend_silence_sec=0):
    """Bytes frames list → WAV in memory (BytesIO). Optionally prepend silence."""
    assert _pyaudio_instance is not None
    raw = b"".join(frames)
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    audio_resampled = _resample_audio(audio_int16, MIC_SAMPLE_RATE, SAMPLE_RATE)
    if prepend_silence_sec > 0:
        sample_width = _pyaudio_instance.get_sample_size(AUDIO_FORMAT)
        silence_len = int(prepend_silence_sec * SAMPLE_RATE) * sample_width
        audio_resampled = np.concatenate([np.zeros(silence_len // sample_width, dtype=np.int16), audio_resampled])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(_pyaudio_instance.get_sample_size(AUDIO_FORMAT))
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_resampled.tobytes())
    buf.seek(0)
    return buf


# -----------------------------------------------------------------------------
# Transcription and LLM
# -----------------------------------------------------------------------------

def transcribe(wav_buffer):
    """Transcribe WAV with Whisper. Returns (text, language_code)."""
    assert _whisper_model is not None
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


def transform_with_llm(raw_text, detected_lang):
    """LLM transform: post-process transcription via Ollama."""
    if not raw_text.strip():
        return raw_text
    print("🔄 LLM transform...")
    t0 = time.time()
    prompt = LLM_TRANSFORM_PROMPT.format(detected_lang=detected_lang, raw_text=raw_text)
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

def _active_window_class():
    """Return lowercase WM_CLASS of the active X11 window, or None on failure."""
    if not shutil.which("xprop"):
        return None
    try:
        root = subprocess.run(
            ["xprop", "-root", "_NET_ACTIVE_WINDOW"],
            capture_output=True, text=True, timeout=0.3,
        ).stdout
    except Exception:
        return None
    parts = root.strip().split()
    if not parts:
        return None
    wid = parts[-1]
    if not wid.startswith("0x"):
        return None
    try:
        wm = subprocess.run(
            ["xprop", "-id", wid, "WM_CLASS"],
            capture_output=True, text=True, timeout=0.3,
        ).stdout
    except Exception:
        return None
    return wm.lower()


def _detect_paste_shortcut():
    """Pick paste shortcut based on active window class. Fallback to default."""
    if not AUTO_DETECT_TERMINAL:
        return DEFAULT_PASTE_SHORTCUT
    wm = _active_window_class()
    if wm is None:
        return DEFAULT_PASTE_SHORTCUT
    if any(cls in wm for cls in TERMINAL_WM_CLASSES):
        return TERMINAL_PASTE_SHORTCUT
    return DEFAULT_PASTE_SHORTCUT


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
        shortcut = _detect_paste_shortcut()
        keyboard.send(shortcut)
        time.sleep(0.1)
        if KEYS_AFTER_PASTE:
            time.sleep(0.05)
            keyboard.send(KEYS_AFTER_PASTE)
        suffix = f' + "{KEYS_AFTER_PASTE.upper()}"' if KEYS_AFTER_PASTE else ""
        print(f'✅ Pasted to active window via "{shortcut.upper()}"{suffix}!')
        if CLIPBOARD_AFTER_PASTE_POLICY == "restore":
            pyperclip.copy(old)
        elif CLIPBOARD_AFTER_PASTE_POLICY == "clear":
            pyperclip.copy("")


# -----------------------------------------------------------------------------
# Process recording (background thread)
# -----------------------------------------------------------------------------

def _process_recorded_frames(frames):
    """Pipeline: frames → resample → WAV → Whisper → optional LLM → paste."""
    assert _pyaudio_instance is not None
    raw = b"".join(frames)
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    audio_int16 = _resample_audio(audio_int16, MIC_SAMPLE_RATE, SAMPLE_RATE)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(_pyaudio_instance.get_sample_size(AUDIO_FORMAT))
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_int16.tobytes())
    buf.seek(0)
    raw_text, lang = transcribe(buf)
    if USE_LLM_TRANSFORM and raw_text.strip():
        final_text = transform_with_llm(raw_text, lang)
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
    duration_sec = len(frames) * CHUNK_SIZE / MIC_SAMPLE_RATE
    print(f"⏹️ Recorded {duration_sec:.1f}s (with {PREBUFFER_SEC}s prebuffer)")

    # Only process recordings longer than 0.7 seconds in total.
    if duration_sec <= 0.7 or len(frames) < MIN_FRAMES:
        print("❌ Recording too short")
        return

    # Simple silence / noise gate: skip very low-energy audio.
    raw = b"".join(frames)
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    audio_int16 = _resample_audio(audio_int16, MIC_SAMPLE_RATE, SAMPLE_RATE)
    if audio_int16.size == 0 or np.max(np.abs(audio_int16)) < SILENCE_AMPLITUDE_THRESHOLD:
        print(f"❌ Audio too quiet / silence, skipping (max={np.max(np.abs(audio_int16))}, threshold={SILENCE_AMPLITUDE_THRESHOLD})")
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


def _format_input_device_config():
    if INPUT_DEVICE_INDEX:
        return f"index {INPUT_DEVICE_INDEX}"
    if INPUT_DEVICE_NAME:
        return f'name contains "{INPUT_DEVICE_NAME}"'
    if AUTO_SELECT_INPUT_DEVICE:
        return "auto-select"
    return "system default"


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
        line(f"     LLM transform: {'ON' if USE_LLM_TRANSFORM else 'OFF'}") + "\n",
        line(f"     Copy to clipboard: {'ON' if COPY_TO_CLIPBOARD else 'OFF'}") + "\n",
        line(f"     Paste to active window: {'ON' if PASTE_TO_ACTIVE_WINDOW else 'OFF'}") + "\n",
        line(f"     Input device: {_format_input_device_config()}") + "\n",
    ]
    if PASTE_TO_ACTIVE_WINDOW:
        if AUTO_DETECT_TERMINAL:
            parts.append(line(f'     Paste shortcut: "{DEFAULT_PASTE_SHORTCUT.upper()}" / "{TERMINAL_PASTE_SHORTCUT.upper()}" in terminals (auto-detect)') + "\n")
        else:
            parts.append(line(f'     Paste shortcut: "{DEFAULT_PASTE_SHORTCUT.upper()}"') + "\n")
        parts.append((line(f'     Keys after paste: "{KEYS_AFTER_PASTE.upper()}"') if KEYS_AFTER_PASTE else line("     Keys after paste: —")) + "\n")
    parts.extend([line("") + "\n", line('     "CTRL+C" to exit') + "\n", "╚" + "═" * w + "╝"])
    return "".join(parts)


def main():
    global _pyaudio_instance, _whisper_model, _prebuffer_deque

    _pyaudio_instance = pyaudio.PyAudio()
    if LIST_AUDIO_DEVICES:
        _print_input_devices()
        return

    print("⏳ Loading Whisper model... (first run may download the model)")
    _whisper_model = WhisperModel(
        WHISPER_MODEL,
        device="cuda",
        compute_type="float16",
    )
    print("✅ Whisper loaded!")

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
