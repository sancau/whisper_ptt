<p align="left">
  <img src="assets/whisper_ptt.png" alt="Whisper-PTT" width="192">
</p>

# Whisper-PTT

**Local voice-to-text · Push-to-talk · Offline.**

Voice-to-text tools shouldn't require blind trust. **Whisper-PTT** is a **single-file push-to-talk** utility: it turns your speech into text locally with Whisper, then optionally polishes it with an LLM pass (Ollama) — cleaning up filler words, fixing grammar, and adding punctuation. Both steps run fully offline; nothing leaves your machine. The whole source is short enough to read over coffee — you can verify exactly where your audio goes. **Hold a hotkey → speak → release** → clean text appears in your active window (or clipboard). That's it.

The pipeline: hold the hotkey, speak, release. Whisper transcribes locally; an optional second pass through an LLM cleans the result before it's pasted into the focused window. Everything is configurable via environment variables: Whisper model, hotkey, LLM model, cleanup prompt, paste vs clipboard-only, or turn off LLM entirely. A prebuffer captures the start of your speech so the first word isn't clipped. One Python file, a handful of dependencies — no build step, no daemon, no config files you didn't ask for.

**Why not on PyPI?**

> *Whisper-PTT is intentionally not on PyPI. The point is that you can read the entire source before running it. `pip install` would undermine that.*

Clone → open → read → run: you can audit the code in a few minutes. Publishing to PyPI would scatter files into `site-packages` and add a layer of abstraction between you and the code — the opposite of "one file, zero blind trust." Same philosophy as other single-file, audit-friendly tools: you see what you run.

---

## Quick start

> **Tested on Windows.** Linux and macOS should work; on Linux the `keyboard` library typically needs root for global hotkeys (`sudo`).
> The pinned `requirements.txt` was generated on Windows with CUDA; on CPU-only or other platforms you may use the minimal install (see step 1).

**1. Clone and install**

```bash
git clone https://github.com/yourname/whisper-ptt.git
cd whisper-ptt
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires **Python 3.9+**. The repo ships a pinned `requirements.txt` for reproducible installs. If you prefer minimal deps or hit conflicts (e.g. CPU-only), install the core ones: `pip install faster-whisper pyaudio keyboard pyperclip requests`. Optional: `pip install python-dotenv` to load config from a `.env` file; without it, only environment variables are used, and if a `.env` file exists the script will print a note to install python-dotenv.

**2. (Optional) Ollama for LLM cleanup**

```bash
# Install: https://ollama.com/download
ollama pull gemma3:12b
```

Skip this and set `WHISPER_PTT_USE_LLM_CLEANUP=false` in `.env` if you only want raw Whisper output.

**3. Configure**

```bash
cp .env.example .env
```

Edit `.env` as needed. Main knobs:

| Variable | What it does | Default |
|----------|--------------|---------|
| `WHISPER_PTT_WHISPER_MODEL` | Whisper model (`base`, `small`, `medium`, `large-v3`, `large-v3-turbo`) | `large-v3` |
| `WHISPER_PTT_WHISPER_DEVICE` | Whisper device: `cuda` or `cpu` | `cuda` |
| `WHISPER_PTT_WHISPER_LANGUAGE` | Whisper language (`en`, `ru`, …) | `en` |
| `WHISPER_PTT_HOTKEY` | Hotkey (`pause`, `f9`, `f10`, `scroll lock`, …) | `pause` |
| `WHISPER_PTT_USE_LLM_CLEANUP` | LLM cleanup | `true` (off: `false`, `0`, `no`, `off`) |
| `WHISPER_PTT_OLLAMA_MODEL` | Ollama model (for LLM cleanup) | `gemma3:12b` |
| `WHISPER_PTT_COPY_TO_CLIPBOARD` | Copy to clipboard | `true` (same on/off values as above) |
| `WHISPER_PTT_PASTE_TO_ACTIVE_WINDOW` | Paste to active window | `true` (same on/off values as above) |
| `WHISPER_PTT_KEYS_AFTER_PASTE` | Keys after paste: key(s) to send (`enter`, `ctrl+enter`, or empty/`none`) | `enter` |

<details>
<summary>All other variables (audio, prebuffer, advanced)</summary>

| Variable | What it does | Default |
|----------|--------------|---------|
| `WHISPER_PTT_WHISPER_COMPUTE_TYPE` | Whisper compute type: `float16`, `int8`, `float32` | `float16` |
| `WHISPER_PTT_WHISPER_INITIAL_PROMPT` | Whisper initial prompt (e.g. language hint) | `English speech.` |
| `WHISPER_PTT_OLLAMA_URL` | Ollama URL | `http://localhost:11434/api/generate` |
| `WHISPER_PTT_LLM_CLEANUP_PROMPT` | LLM cleanup prompt; placeholders `{detected_lang}`, `{raw_text}` | built-in |
| `WHISPER_PTT_SAMPLE_RATE` | Sample rate | `16000` |
| `WHISPER_PTT_CHUNK_SIZE` | Chunk size | `1024` |
| `WHISPER_PTT_PREBUFFER_SEC` | Prebuffer (sec; captures first word) | `0.5` |
| `WHISPER_PTT_PADDING_SEC` | Padding (sec; silence before Whisper) | `0.2` |
| `WHISPER_PTT_MIN_FRAMES` | Min frames (skip accidental taps) | `5` |

</details>

**4. Run**

```bash
python whisper_ptt.py
```

First run may download the Whisper model (size depends on the model you chose). No CUDA? Set `WHISPER_PTT_WHISPER_DEVICE=cpu` in `.env`.

**5. Use**

Hold **Pause** (or your hotkey) → speak → release. Text is pasted into the active window (and Enter is sent if enabled). Exit with **Esc** or Ctrl+C.

---

## Use cases

- **Chats / AI assistants** — speak a message, get it pasted and sent (Cursor, Slack, Discord).
- **Any text field** — focus the field, hold hotkey, speak, release.
- **Offline / privacy** — all processing local; no data sent to third parties.

---

## Project layout

```
whisper-ptt/
  whisper_ptt.py     # single entrypoint
  requirements.txt   # pinned dependencies (or use minimal: see Quick start)
  .env.example       # config template
  .env               # your config (optional)
  README.md
```

One script, clear sections (config, prebuffer, transcription, LLM, paste, hotkeys). Easy to audit and adapt.

---

## Dependencies

The included `requirements.txt` uses **strict pins** (`==`) so that `pip install -r requirements.txt` gives a reproducible environment. For an open-source app this is a common choice: "clone and run" behaves the same for everyone. If you prefer looser versions (e.g. `faster-whisper>=1.0`, `pyaudio>=0.2.12`) for easier integration with other projects, keep a minimal list of direct deps with `>=` and omit the lockfile, or maintain both a minimal `requirements.in` and a generated pinned `requirements.txt`. The current file was frozen from a Windows + CUDA environment; CPU-only or other platforms can install the core packages by hand (see Quick start step 1).

---

## License

MIT — see [LICENSE](LICENSE).
