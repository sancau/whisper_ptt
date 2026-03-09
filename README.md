<div align="center">

# 🎙 Whisper-PTT

**Local voice-to-text · Push-to-talk · Offline**

Hold a hotkey → speak → release → text appears. That's it.

<img src="assets/demo.gif" alt="Whisper-PTT demo" width="820">

</div>


## Why?

Voice-to-text tools shouldn't require blind trust. **Whisper-PTT** is a **single-file push-to-talk** utility: it turns your speech into text locally with Whisper, then (optionally) runs a cleanup pass with a local LLM (Ollama) — cleaning up filler words, fixing grammar, and adding punctuation. Both steps run fully offline; nothing leaves your machine. The whole source is short enough to read over coffee — you can verify exactly where your audio goes. **Core behavior:** hold a hotkey → speak → release → raw Whisper text appears in your active window (or clipboard). LLM cleanup is an extra layer you can enable if you want more polished output.

**PyPI?**

> *Whisper-PTT is intentionally not on PyPI. The point is that you can read the entire source before running it. `pip install` would undermine that.*

Clone → open → read → run: you can audit the code in a few minutes. Publishing to PyPI would scatter files into `site-packages` and add a layer of abstraction between you and the code — the opposite of "one file, zero blind trust."

---

## Platform files

Whisper-PTT requires GPU acceleration — there is no CPU-only mode. The tool ships as **two standalone scripts**, one per platform. Each file is a complete, self-contained tool with zero shared imports between them. This is intentional: the goal is minimal code you can audit in one sitting, not a framework with layers of abstraction. Different platforms need different Whisper backends, different paste mechanisms, and different audio quirks — splitting keeps each file focused and short.

| Platform | Script | Whisper backend | Accelerator |
|----------|--------|-----------------|-------------|
| **Windows / Linux** | `whisper_ptt_cuda.py` | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) | NVIDIA CUDA |
| **macOS** | `whisper_ptt_apple_silicon.py` | [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (MLX) | Apple Silicon Metal |

Pick the file for your platform and ignore the other one. They share the same `.env` config format and the same workflow (prebuffer → Whisper → optional LLM cleanup → paste).

---

## Quick start

### Windows / Linux (NVIDIA CUDA)

Requires an **NVIDIA GPU** with CUDA support.

```bash
git clone https://github.com/sancau/whisper-ptt.git
cd whisper-ptt
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements-cuda.txt
cp .env.example-cuda .env       # edit as needed
python whisper_ptt_cuda.py
```

On Linux, the `keyboard` library typically needs root for global hotkeys (`sudo python whisper_ptt_cuda.py`).

### macOS Apple Silicon (M1/M2/M3/M4)

Requires an **Apple Silicon** Mac.

```bash
git clone https://github.com/sancau/whisper-ptt.git
cd whisper-ptt
python -m venv venv
source venv/bin/activate
pip install -r requirements-apple-silicon.txt
cp .env.example-apple-silicon .env   # edit as needed
python whisper_ptt_apple_silicon.py
```

On first run, macOS will prompt for **Accessibility** and **Input Monitoring** permissions for your terminal app so it can listen for global hotkeys and send paste keystrokes. Grant those, and run **without** `sudo`. The first run downloads the Whisper model from HuggingFace (mlx-community). Inference runs on Metal — no CUDA needed.

### (Optional) Ollama for LLM cleanup (disabled by default)

```bash
# Install: https://ollama.com/download
ollama pull gemma3:12b
```

By default, **LLM cleanup is off** — you get **raw Whisper text**, and you don't need Ollama installed. Turn it on later if you want extra polish.

---

## Configuration

All config is via `WHISPER_PTT_*` environment variables or a `.env` file. Both scripts read the same variable names.

```bash
cp .env.example-cuda .env                  # Windows / Linux
cp .env.example-apple-silicon .env         # macOS Apple Silicon
```

### Main settings

| Variable | What it does | Default |
|----------|--------------|---------|
| `WHISPER_PTT_WHISPER_MODEL` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`) | `large-v3` (CUDA), `large-v3-turbo` (Apple Silicon) |
| `WHISPER_PTT_WHISPER_LANGUAGE` | Whisper language code (`en`, `ru`, `de`, `fr`, …) | `en` |
| `WHISPER_PTT_HOTKEY` | Hotkey (hold to record). Combos like `alt+f12` also work. | `alt` (CUDA), `option` (Apple Silicon) |
| `WHISPER_PTT_USE_LLM_CLEANUP` | LLM cleanup on/off | `false` |
| `WHISPER_PTT_OLLAMA_MODEL` | Ollama model for cleanup | `gemma3:12b` |
| `WHISPER_PTT_COPY_TO_CLIPBOARD` | Copy result to clipboard | `true` |
| `WHISPER_PTT_PASTE_TO_ACTIVE_WINDOW` | Paste into the focused window | `true` |
| `WHISPER_PTT_CLIPBOARD_AFTER_PASTE_POLICY` | After paste: `restore` (default), `clear`, or `preserve` | `restore` |
| `WHISPER_PTT_KEYS_AFTER_PASTE` | Key(s) to send after paste (`enter`, `ctrl+enter`, or `none`) | `enter` |

<details>
<summary>All other variables (audio, prebuffer, advanced)</summary>

| Variable | What it does | Default |
|----------|--------------|---------|
| `WHISPER_PTT_WHISPER_INITIAL_PROMPT` | Whisper initial prompt (language hint) | `English speech.` |
| `WHISPER_PTT_OLLAMA_URL` | Ollama API URL | `http://localhost:11434/api/generate` |
| `WHISPER_PTT_LLM_CLEANUP_PROMPT` | Custom LLM prompt; placeholders `{detected_lang}`, `{raw_text}` | built-in |
| `WHISPER_PTT_SAMPLE_RATE` | Sample rate (Hz) | `16000` |
| `WHISPER_PTT_CHUNK_SIZE` | Audio chunk size | `1024` |
| `WHISPER_PTT_PREBUFFER_SEC` | Prebuffer duration (captures the first word) | `0.5` |
| `WHISPER_PTT_PADDING_SEC` | Silence padding before Whisper | `0.2` |
| `WHISPER_PTT_MIN_FRAMES` | Min frames to process (skip accidental taps) | `5` |
| `WHISPER_PTT_SILENCE_AMPLITUDE` | Simple silence gate: if max int16 amplitude is below this, treat audio as silence and skip transcription | `750` |

</details>

---

## Usage

Default hotkey: **Alt** (Windows/Linux) or **Option** (macOS). Hold → speak → release. Text is pasted into the active window (and Enter is sent if configured). Exit with **Esc** or Ctrl+C.

### Use cases

- **Chats / AI assistants** — speak a message, get it pasted and sent (Cursor, Slack, Discord).
- **Any text field** — focus the field, hold hotkey, speak, release.
- **Offline / privacy** — all processing local; no data sent to third parties.
- **(Optional) LLM cleanup** — when enabled, use a local Ollama model to:
  - turn spoken notes into **documentation-style** or email-ready text,
  - do **on-the-go translation / bilingual workflows** (e.g. speak in one language, get output in another via a custom prompt),
  - clean up prompts before sending them to other **chat/AI assistants**.

---

## Project layout

```
whisper-ptt/
  whisper_ptt_cuda.py              # Windows / Linux  (faster-whisper + NVIDIA CUDA)
  whisper_ptt_apple_silicon.py     # macOS  (mlx-whisper + Apple Silicon Metal)
  requirements-cuda.txt            # pinned deps — CUDA
  requirements-apple-silicon.txt   # deps — Apple Silicon
  .env.example-cuda                # config template — CUDA
  .env.example-apple-silicon       # config template — Apple Silicon
  .env                             # your config (git-ignored)
  README.md
```

Each script is self-contained with clear sections (config, prebuffer, transcription, LLM, paste, hotkeys). One file per platform — easy to audit and adapt.

---

## License

MIT — see [LICENSE](LICENSE).
