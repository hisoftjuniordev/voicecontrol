# VoiceChat — Setup & Pre-flight for Claude Code

100 % local, no API keys. Mic → Whisper (STT) → Ollama (LLM) → text on screen.

---

## What's in this repo

| File | Purpose |
|------|---------|
| `voice_chat.py` | v1 — basic hands-free voice chat |
| `voice_chat_v2.py` | v2 — sessions, resource monitor, agent mode |
| `run_voice_chat.bat` | Launches v1 windowless via `pythonw` |
| `run_voice_chat_v2.bat` | Launches v2 windowless via `pythonw` |
| `make_hermes_64k.bat` | Creates a 64K-context Ollama model for v2 agent mode |

---

## Pre-flight checklist — run these checks before launching any .bat

### 1. Python

```powershell
python --version
pythonw --version
```

- Needs Python 3.10+.
- `pythonw.exe` must exist (ships with the standard Windows installer; needed by the .bat files).
- If missing: install from python.org — tick "Add Python to PATH".

### 2. Python packages

```powershell
pip show faster-whisper sounddevice numpy psutil pywin32
```

All five must be present. Install any that are missing:

```powershell
pip install faster-whisper sounddevice numpy psutil pywin32
```

> `psutil` and `pywin32` are only used by v2. v1 only needs `faster-whisper sounddevice numpy`.

### 3. Ollama

```powershell
ollama --version
```

- If missing: download from https://ollama.com and install.
- After install, pull at least one model (see §4).
- The app auto-starts `ollama serve` if it isn't already running, so no manual start is needed.

**NVIDIA GPU acceleration** — Ollama detects CUDA automatically if NVIDIA drivers are current.
Check driver status with:
```powershell
nvidia-smi
```
If `nvidia-smi` fails, update drivers from https://www.nvidia.com/drivers before running Ollama.

### 4. Ollama models

```powershell
ollama list
```

The app picks from these preferred models in order:
`llama3.2:3b` → `gemma4:e4b` → `qwen3.5:4b` → `phi4-mini:latest` → `llama3.1:8b` → `qwen3:8b`

Pull whichever fits the VRAM:

```powershell
# ~2 GB VRAM (safe for most Nvidia cards)
ollama pull llama3.2:3b

# ~3-4 GB VRAM (better quality)
ollama pull qwen3.5:4b

# ~5-6 GB VRAM
ollama pull llama3.1:8b
```

If none of the preferred models are found, the app will still run but the model dropdown will be empty — pick one manually in Settings.

### 5. Microphone

Confirm a microphone is connected and Windows sees it:

```powershell
python -c "import sounddevice as sd; print([d['name'] for d in sd.query_devices() if d['max_input_channels'] > 0])"
```

Must return at least one device. If the list is empty, check Windows Sound settings → Input.

### 6. (v2 only) Agent mode — 64K context model

Agent mode requires a model with a 64K context window. Create one from any already-pulled model:

```bat
make_hermes_64k.bat qwen3.5:4b
```

This creates `qwen3.5-4b-64k` in Ollama. Then in the app go to Settings → Agent model and select it.

> **VRAM note (NVIDIA):** 64K context on a 4B model needs ~4 GB VRAM with Flash Attention + q8 KV cache. On cards with 6 GB+ this is fine. On 4 GB cards use `llama3.2:3b` instead.

---

## Typical first-run sequence on a fresh Nvidia build

```powershell
# 1. Install Python (if needed) — from python.org, tick "Add to PATH"

# 2. Install packages
pip install faster-whisper sounddevice numpy psutil pywin32

# 3. Install Ollama — from ollama.com

# 4. Pull a model
ollama pull llama3.2:3b

# 5. Run the app
# Double-click run_voice_chat_v2.bat   (or run_voice_chat.bat for v1)
```

After first launch, Whisper downloads its speech model automatically (~150 MB for `base`).
The app's Settings panel lets you change the Whisper model, Ollama model, mic, language, and more.

---

## Whisper model guide

| Model | Speed (CPU) | Accuracy | Download size |
|-------|-------------|----------|---------------|
| `tiny` | ~0.5 s | low | ~75 MB |
| `base` | ~1.3 s | good | ~150 MB |
| `small` | ~5 s | better | ~480 MB |
| `medium` | ~15 s | high | ~1.5 GB |
| `large-v3-turbo` | ~30 s | best | ~3 GB |

Default is `base`. Whisper always runs on **CPU** regardless of GPU.

---

## Session data (v2)

- Settings saved to: `voice_chat_v2_data/settings.json`
- Session transcripts saved to: `voice_chat_sessions/<date>_<time>_session.md`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| App opens and closes immediately | Run `python voice_chat_v2.py` in a terminal to see the error |
| "Could not open microphone" | Select a mic in Settings, or check Windows Sound → Input |
| "No Ollama model selected" | Open Settings, click Refresh models, select one |
| Ollama not starting | Run `ollama serve` manually in a terminal; check firewall on port 11434 |
| Very slow responses | GPU not being used — check `nvidia-smi`, update drivers |
| App won't start (second instance) | A copy is already running; check the taskbar / system tray |
