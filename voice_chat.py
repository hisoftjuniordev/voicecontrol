"""
Hands-free Voice Chat  -  100 % local, no API keys, no subscriptions
===================================================================
  headset mic -> voice-activity detection (Silero) -> faster-whisper (local STT)
              -> Ollama (local LLM, streamed) -> big readable text on screen

Start it once (run_voice_chat.bat) and just talk. It starts recording when you speak, sends the
question when you have been silent for N seconds (default 2), and prints the answer.
You never need the keyboard.

Voice commands (say them on their own):
    "pause listening" / "resume listening"   - ignore everything until resumed
    "new chat"                               - forget the conversation so far
    "stop" / "cancel"                        - abort the answer that is being written

Optional wake word (Settings): only act on sentences that start with e.g. "computer, ...".
Replies are text only (no speech output), so the mic can stay open the whole time.

Install: pip install faster-whisper sounddevice numpy      (+ Ollama running with a model pulled)
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "voice_chat_data"
SETTINGS_FILE = DATA_DIR / "settings.json"

OLLAMA = "http://127.0.0.1:11434"
SR = 16000
FRAME = 512                       # Silero frame size at 16 kHz (32 ms)
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo"]
LANGUAGES = ["en", "sl", "auto", "hr", "de", "it", "es", "fr"]
PREFERRED_MODELS = ["llama3.2:3b", "gemma4:e4b", "qwen3.5:4b", "phi4-mini:latest", "llama3.1:8b", "qwen3:8b"]

DEFAULT_PROMPT = ("You are a hands-free voice assistant for someone who is working out and cannot touch the keyboard. "
                  "Their words come from speech recognition, so expect small transcription errors and infer what they meant. "
                  "Answer directly and briefly (usually under 80 words), in plain text without markdown, "
                  "so it is easy to read from a distance. Give longer answers only when explicitly asked.")

DEFAULTS = {
    "ollama_model": "",
    "whisper_model": "base",    # ~1.3 s per sentence on CPU; "small" is more accurate but ~4x slower
    "language": "en",
    "mic": "",
    "silence_s": 2.0,           # silence that ends an utterance
    "sensitivity": 0.5,         # higher = picks up quieter speech
    "wake_enabled": False,
    "wake_word": "computer",
    "system_prompt": DEFAULT_PROMPT,
    "font_size": 22,
    "keep_alive": "30m",        # how long Ollama keeps the model in memory
    "show_settings": True,
}

PAUSE_CMDS = {"pause listening", "stop listening", "pause", "go to sleep"}
RESUME_CMDS = {"resume listening", "start listening", "resume", "wake up", "listen"}
CLEAR_CMDS = {"new chat", "clear chat", "new conversation", "clear conversation", "reset conversation"}
STOP_CMDS = {"stop", "cancel", "stop it", "stop generating", "never mind", "nevermind"}
# Whisper tends to invent these on short noisy clips
HALLUCINATIONS = {"you", "thank you", "thanks", "thanks for watching", "thank you for watching", "bye", "okay", "uh", "um", "hmm"}


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
def load_settings():
    s = dict(DEFAULTS)
    try:
        s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8-sig")))
    except (OSError, ValueError):
        pass
    return s


def save_settings(s):
    try:
        DATA_DIR.mkdir(exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(s, indent=2), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# voice-activity detection / utterance segmentation
# ---------------------------------------------------------------------------
class Segmenter:
    """Feed 512-sample frames; returns a finished utterance (float32 array) after enough silence."""

    def __init__(self, sensitivity=0.5, silence_s=2.0, max_s=60.0):
        from faster_whisper.vad import get_vad_model     # Silero VAD bundled with faster-whisper
        self.vad = get_vad_model()
        self.sensitivity = sensitivity
        self.silence_s = silence_s
        self.max_s = max_s
        self.pre = deque(maxlen=12)                       # ~0.4 s pre-roll so first syllable isn't cut
        self.win = np.zeros(FRAME * 4, dtype=np.float32)  # sliding window gives Silero some context
        self.active = False
        self.buf, self.run, self.voiced, self.silent = [], 0, 0, 0
        self.prob = 0.0

    def _prob(self, frame):
        self.win = np.concatenate([self.win[FRAME:], frame])
        return float(np.ravel(self.vad(self.win))[-1])

    def reset(self):
        self.active = False
        self.buf, self.run, self.voiced, self.silent = [], 0, 0, 0
        self.pre.clear()

    def feed(self, frame):
        p = self.prob = self._prob(frame)
        thr = 1.0 - self.sensitivity
        if not self.active:
            self.pre.append(frame)
            self.run = self.run + 1 if p >= thr else 0
            if self.run >= 2:                             # two consecutive speech frames = start
                self.active, self.buf = True, list(self.pre)
                self.voiced, self.silent = self.run, 0
            return None
        self.buf.append(frame)
        if p >= max(0.1, thr - 0.15):                     # hysteresis: quiet word endings still count as speech
            self.voiced += 1
            self.silent = 0
        else:
            self.silent += 1
        secs = len(self.buf) * FRAME / SR
        if self.silent * FRAME / SR >= self.silence_s or secs >= self.max_s:
            keep = len(self.buf) - max(0, self.silent - 10)   # trim trailing silence, keep 0.3 s
            utt = np.concatenate(self.buf[:keep])
            enough = self.voiced >= 8
            self.reset()
            return utt if enough else None
        return None


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------
def ollama_up(timeout=2):
    try:
        urllib.request.urlopen(OLLAMA + "/api/tags", timeout=timeout).read()
        return True
    except Exception:  # noqa: BLE001
        return False


_spawned = None                    # the `ollama serve` process WE started (never the tray app's)
_ollama_lock = threading.Lock()


def ensure_ollama(log):
    """Makes sure a server answers on :11434. Starts at most ONE `ollama serve`, ever."""
    global _spawned
    if ollama_up():
        return True
    with _ollama_lock:                                    # threads that arrive together queue up here...
        if ollama_up():                                   # ...and find it already running
            return True
        if _spawned is None or _spawned.poll() is not None:
            log("Ollama is not running - starting 'ollama serve'...")
            try:
                flags = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW
                _spawned = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, creationflags=flags)
            except OSError as e:
                log(f"Could not start Ollama: {e}")
                return False
        for _ in range(40):                               # a server we started is still booting: just wait
            if ollama_up():
                return True
            time.sleep(0.5)
    log("Ollama did not come up within 20 s")
    return False


def unload_model(name):
    """Tell Ollama to drop a model from RAM/VRAM now (keep_alive 0)."""
    if not name:
        return
    try:
        req = urllib.request.Request(OLLAMA + "/api/generate", data=json.dumps({"model": name, "keep_alive": 0}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:  # noqa: BLE001 - server already gone is fine
        pass


def stop_spawned_ollama():
    """On exit: stop the server only if this app started it. A running tray app is left alone."""
    global _spawned
    if _spawned is not None and _spawned.poll() is None:
        _spawned.terminate()
    _spawned = None


_instance_handle = None


def single_instance():
    """False if another copy of this app (from ANY folder) is already running."""
    global _instance_handle
    if sys.platform != "win32":
        return True
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _instance_handle = k32.CreateMutexW(None, False, "Local\\LocalVoiceChatHandsFree")
    return ctypes.get_last_error() != 183                 # ERROR_ALREADY_EXISTS


def list_models():
    try:
        data = json.loads(urllib.request.urlopen(OLLAMA + "/api/tags", timeout=5).read())
        return sorted(m["name"] for m in data.get("models", []))
    except Exception:  # noqa: BLE001
        return []


def pick_default_model(models):
    for p in PREFERRED_MODELS:
        if p in models:
            return p
    usable = [m for m in models if "embed" not in m]
    return usable[0] if usable else ""


def visible_text(raw):
    """Remove <think>...</think> blocks (also an unfinished one) from streamed output."""
    return re.sub(r"<think>.*?(</think>|$)", "", raw, flags=re.S).lstrip()


def chat_stream(model, messages, keep_alive, abort, on_raw):
    """Streams a reply from Ollama. Calls on_raw(full_text_so_far, thinking_bool). Returns the raw text."""
    def post(with_think_flag):
        body = {"model": model, "messages": messages, "stream": True, "keep_alive": keep_alive}
        if with_think_flag:
            body["think"] = False                         # skip hidden reasoning on thinking models: faster answers
        req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=300)

    try:
        resp = post(True)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        if e.code == 400 and "think" in detail.lower():
            resp = post(False)
        else:
            raise RuntimeError(f"Ollama error {e.code}: {detail[:200]}") from e
    raw, thinking = "", False
    with resp:
        for line in resp:
            if abort.is_set():
                break
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("error"):
                raise RuntimeError(d["error"])
            msg = d.get("message", {})
            if msg.get("thinking"):
                thinking = True
            if msg.get("content"):
                raw += msg["content"]
                thinking = False
            on_raw(raw, thinking)
            if d.get("done"):
                break
    return raw


# ---------------------------------------------------------------------------
# Engine: mic frames -> utterances -> text -> Ollama
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, settings, emit):
        self.s = settings
        self.emit = emit                                  # emit(kind, **payload); must be thread-safe
        self.seg = None
        self.frames = queue.Queue(maxsize=600)            # ~19 s of audio backlog at most
        self.utts = queue.Queue()
        self.chats = queue.Queue()
        self.history = []
        self.whisper = None
        self.whisper_ready = threading.Event()
        self.paused = False
        self.stt_busy = 0
        self.generating = False
        self.abort = threading.Event()
        self._last_state = None
        self._stopping = False
        self._wlock = threading.Lock()
        self._load_lock = threading.Lock()                # one Whisper load at a time, never two models in RAM
        self._loaded_llm = None                           # the Ollama model this app last used

    # ----- state banner -----
    def state(self):
        if not self.whisper_ready.is_set():
            return "loading"
        if self.paused:
            return "paused"
        if self.seg and self.seg.active:
            return "recording"
        if self.stt_busy:
            return "transcribing"
        if self.generating:
            return "thinking"
        return "listening"

    def _update_state(self):
        st = self.state()
        if st != self._last_state:
            self._last_state = st
            self.emit("state", name=st)

    # ----- lifecycle -----
    def start(self):
        self.seg = Segmenter(self.s["sensitivity"], self.s["silence_s"])
        for fn in (self._vad_loop, self._stt_loop, self._chat_loop):
            threading.Thread(target=fn, daemon=True).start()
        threading.Thread(target=self.load_whisper, daemon=True).start()
        threading.Thread(target=lambda: ensure_ollama(lambda m: self.emit("system", text=m)), daemon=True).start()

    def stop(self):
        self._stopping = True
        self.abort.set()

    def release_models(self):
        """Exit cleanup: free the Ollama model this app loaded, and the server if we started it."""
        unload_model(self._loaded_llm)
        stop_spawned_ollama()

    def load_whisper(self):
        from faster_whisper import WhisperModel
        with self._load_lock:                             # rapid dropdown changes queue up; last choice wins
            name = self.s["whisper_model"]
            self.whisper_ready.clear()
            self._update_state()
            self.emit("system", text=f"Loading speech model '{name}' (first run downloads it)...")
            try:
                model = WhisperModel(name, device="cpu", compute_type="int8")
                with self._wlock:
                    self.whisper = model                  # old model is dropped here
                self.whisper_ready.set()
                self.emit("system", text=f"Speech model ready ({name}). Just start talking.")
            except Exception as e:  # noqa: BLE001
                self.emit("error", text=f"Could not load speech model: {e}")
            self._update_state()

    # ----- audio in -----
    def feed(self, block):
        try:
            self.frames.put_nowait(block)
        except queue.Full:
            pass

    def _vad_loop(self):
        n = 0
        while not self._stopping:
            try:
                frame = self.frames.get(timeout=0.5)
            except queue.Empty:
                continue
            self.seg.sensitivity = self.s["sensitivity"]
            self.seg.silence_s = self.s["silence_s"]
            if len(frame) != FRAME:
                continue
            utt = self.seg.feed(frame)
            n += 1
            if n % 3 == 0:
                self.emit("level", p=self.seg.prob)
            if utt is not None:
                self.stt_busy += 1
                self.utts.put(utt)
            self._update_state()

    # ----- speech to text -----
    def transcribe(self, audio):
        lang = None if self.s["language"] == "auto" else self.s["language"]
        with self._wlock:
            segs, _ = self.whisper.transcribe(audio, language=lang, beam_size=1, vad_filter=True,
                                              condition_on_previous_text=False, without_timestamps=True)
            parts = [x.text.strip() for x in segs if not (x.no_speech_prob > 0.6 and x.avg_logprob < -1.0)]
        text = " ".join(p for p in parts if p).strip()
        if len(audio) / SR < 3 and re.sub(r"[^\w ]", "", text.lower()).strip() in HALLUCINATIONS:
            return ""
        return text

    def _stt_loop(self):
        while not self._stopping:
            try:
                audio = self.utts.get(timeout=0.5)
            except queue.Empty:
                continue
            self.whisper_ready.wait()
            self._update_state()
            try:
                text = self.transcribe(audio)
            except Exception as e:  # noqa: BLE001
                text = ""
                self.emit("error", text=f"Transcription failed: {e}")
            self.stt_busy -= 1
            self._update_state()
            if text:
                self.handle_text(text)

    # ----- text -> command or chat -----
    @staticmethod
    def _norm(text):
        return " ".join(re.findall(r"\w+", text.lower()))

    def _strip_wake(self, text):
        wake = self._norm(self.s["wake_word"])
        words = re.findall(r"\S+", text)
        for i, w in enumerate(words[:4]):
            if self._norm(w) == wake:
                return " ".join(words[i + 1:]).lstrip(",.:;!? ")
        return None

    def handle_text(self, text):
        if self.s["wake_enabled"]:
            rest = self._strip_wake(text)
            if rest is None:
                self.emit("ignored", text=text)
                return
            text = rest
            if not text:
                self.emit("system", text="(listening...)")
                return
        cmd = self._norm(text)
        if cmd in PAUSE_CMDS:
            self.paused = True
            self.emit("system", text="Paused. Say 'resume listening' to continue.")
        elif cmd in RESUME_CMDS:
            self.paused = False
            self.emit("system", text="Listening again.")
        elif self.paused:
            self.emit("ignored", text=text)
        elif cmd in CLEAR_CMDS:
            self.history.clear()
            self.emit("clear")
        elif cmd in STOP_CMDS:
            if self.generating:
                self.abort.set()
            else:
                self.emit("ignored", text=text)
        else:
            self.emit("user", text=text)
            self.chats.put(text)
        self._update_state()

    # ----- chat -----
    def ask(self, text):
        """Used by the typed-input box."""
        self.emit("user", text=text)
        self.chats.put(text)

    def _chat_loop(self):
        while not self._stopping:
            try:
                text = self.chats.get(timeout=0.5)
            except queue.Empty:
                continue
            model = self.s["ollama_model"]
            if not model:
                self.emit("error", text="No Ollama model selected (Settings).")
                continue
            if self._loaded_llm and self._loaded_llm != model:
                unload_model(self._loaded_llm)            # switched models: don't leave the old one in RAM
            self._loaded_llm = model
            self.generating = True
            self.abort.clear()
            self._update_state()
            self.history.append({"role": "user", "content": text})
            messages = [{"role": "system", "content": self.s["system_prompt"]}] + self.history[-16:]
            self.emit("ai_start")
            last = [0.0]

            def on_raw(raw, thinking):
                now = time.time()
                if now - last[0] > 0.06:
                    last[0] = now
                    self.emit("ai_set", text=visible_text(raw), thinking=thinking)

            try:
                if not ensure_ollama(lambda m: self.emit("system", text=m)):
                    raise RuntimeError("Ollama is not reachable")
                raw = chat_stream(model, messages, self.s["keep_alive"], self.abort, on_raw)
                reply = visible_text(raw)
                self.emit("ai_set", text=reply, thinking=False)
                self.history.append({"role": "assistant", "content": reply or "(stopped)"})
                self.emit("ai_end", stopped=self.abort.is_set())
            except Exception as e:  # noqa: BLE001
                self.history.pop()                        # drop the question that failed
                self.emit("ai_end", stopped=True)
                self.emit("error", text=str(e))
            self.generating = False
            self._update_state()


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
BG, PANEL, FG, MUTED = "#0b1220", "#162033", "#f1f5f9", "#94a3b8"
BANNER = {
    "loading": ("LOADING SPEECH MODEL...", "#a78bfa"),
    "listening": ("LISTENING - just talk", "#22c55e"),
    "recording": ("HEARING YOU...", "#ef4444"),
    "transcribing": ("TRANSCRIBING...", "#f59e0b"),
    "thinking": ("THINKING...", "#38bdf8"),
    "paused": ("PAUSED - say \"resume listening\"", "#64748b"),
}


def run_gui():
    import tkinter as tk
    from tkinter import ttk

    import sounddevice as sd
    from tkinter import messagebox

    if not single_instance():                             # a 2nd copy would double RAM/CPU and fight over the mic
        r = tk.Tk()
        r.withdraw()
        messagebox.showinfo("Voice Chat", "Voice Chat is already running (look for its window or taskbar icon).")
        return

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001
            pass

    settings = load_settings()
    evq = queue.Queue()
    engine = Engine(settings, lambda kind, **kw: evq.put((kind, kw)))

    root = tk.Tk()
    root.title("Voice Chat - local Whisper + Ollama")
    root.geometry("1000x760")
    root.configure(bg=BG)

    st = ttk.Style()
    st.theme_use("clam")
    st.configure(".", background=BG, foreground=FG, fieldbackground=PANEL, bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL)
    st.configure("TLabel", background=BG, foreground=FG)
    st.configure("TCheckbutton", background=BG, foreground=FG)
    st.configure("TButton", background=PANEL, foreground=FG, padding=6)
    st.map("TButton", background=[("active", "#22304a")])
    st.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG, arrowcolor=FG)
    st.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", FG)])
    st.configure("TEntry", fieldbackground=PANEL, foreground=FG, insertcolor=FG)
    st.configure("Horizontal.TProgressbar", background="#22c55e", troughcolor=PANEL)
    st.configure("Horizontal.TScale", background="#38bdf8", troughcolor="#334155", bordercolor="#334155")
    root.option_add("*TCombobox*Listbox.background", PANEL)
    root.option_add("*TCombobox*Listbox.foreground", FG)

    banner = tk.Label(root, text="", font=("Segoe UI", 26, "bold"), bg="#64748b", fg="white", pady=10)
    banner.pack(fill="x")
    level = ttk.Progressbar(root, maximum=100)
    level.pack(fill="x", padx=8, pady=(4, 0))

    fs = {"size": int(settings["font_size"])}
    chat = tk.Text(root, wrap="word", bg=BG, fg=FG, bd=0, padx=16, pady=10, state="disabled", cursor="arrow",
                   font=("Segoe UI", fs["size"]), spacing3=8, height=4)   # small requested height: lets the bottom bar fit
    chat.pack(fill="both", expand=True)

    def apply_font():
        base = ("Segoe UI", fs["size"])
        chat.config(font=base)
        chat.tag_config("you", foreground="#7dd3fc", font=("Segoe UI", fs["size"], "bold"))
        chat.tag_config("ai", foreground="#f1f5f9")
        chat.tag_config("sys", foreground=MUTED, font=("Segoe UI", max(10, fs["size"] - 8), "italic"))
        chat.tag_config("err", foreground="#fca5a5", font=("Segoe UI", max(10, fs["size"] - 6)))
        chat.tag_config("ign", foreground="#475569", font=("Segoe UI", max(10, fs["size"] - 10), "italic"))

    apply_font()

    def put(text, tag):
        chat.config(state="normal")
        chat.insert("end", text + "\n", tag)
        chat.config(state="disabled")
        chat.see("end")

    # ----- bottom: typed input + buttons
    bar = ttk.Frame(root)
    bar.pack(fill="x", padx=8, pady=6)
    typed = ttk.Entry(bar, font=("Segoe UI", 13))
    typed.pack(side="left", fill="x", expand=True, ipady=4)

    def send_typed(_=None):
        t = typed.get().strip()
        if t:
            typed.delete(0, "end")
            engine.ask(t)

    typed.bind("<Return>", send_typed)
    btn_pause = ttk.Button(bar, text="Pause")
    btn_pause.pack(side="left", padx=(8, 0))
    ttk.Button(bar, text="New chat", command=lambda: (engine.history.clear(), clear_chat())).pack(side="left", padx=(6, 0))
    ttk.Button(bar, text="Stop answer", command=lambda: engine.abort.set()).pack(side="left", padx=(6, 0))
    ttk.Button(bar, text="A-", width=3, command=lambda: set_font(-2)).pack(side="left", padx=(6, 0))
    ttk.Button(bar, text="A+", width=3, command=lambda: set_font(2)).pack(side="left", padx=(2, 0))
    btn_set = ttk.Button(bar, text="Settings")
    btn_set.pack(side="left", padx=(6, 0))

    def set_font(d):
        fs["size"] = max(12, min(60, fs["size"] + d))
        settings["font_size"] = fs["size"]
        save_settings(settings)
        apply_font()

    def clear_chat():
        chat.config(state="normal")
        chat.delete("1.0", "end")
        chat.config(state="disabled")

    def toggle_pause():
        engine.paused = not engine.paused
        put("Paused." if engine.paused else "Listening again.", "sys")
        engine._update_state()

    btn_pause.config(command=toggle_pause)

    # ----- settings panel
    sp = ttk.Frame(root)
    v = {k: tk.StringVar(value=str(settings[k])) for k in ("ollama_model", "whisper_model", "language", "mic", "wake_word")}
    v_wake = tk.BooleanVar(value=settings["wake_enabled"])
    v_sil = tk.DoubleVar(value=settings["silence_s"])
    v_sens = tk.DoubleVar(value=settings["sensitivity"])

    def mic_names():
        out = []
        try:
            apis = sd.query_hostapis()
            for d in sd.query_devices():
                if d["max_input_channels"] > 0 and apis[d["hostapi"]]["name"] == "MME":
                    out.append(d["name"])
        except Exception:  # noqa: BLE001
            pass
        return out

    def row(r, c, label, widget):
        ttk.Label(sp, text=label).grid(row=r, column=c * 2, sticky="w", padx=(8, 4), pady=3)
        widget.grid(row=r, column=c * 2 + 1, sticky="ew", padx=(0, 8), pady=3)

    cb_model = ttk.Combobox(sp, textvariable=v["ollama_model"], values=list_models(), width=34)
    cb_mic = ttk.Combobox(sp, textvariable=v["mic"], values=mic_names(), width=34, state="readonly")
    cb_wh = ttk.Combobox(sp, textvariable=v["whisper_model"], values=WHISPER_MODELS, width=16, state="readonly")
    cb_lang = ttk.Combobox(sp, textvariable=v["language"], values=LANGUAGES, width=16)
    row(0, 0, "Ollama model", cb_model)
    row(1, 0, "Microphone", cb_mic)
    row(0, 1, "Speech model", cb_wh)
    row(1, 1, "Language", cb_lang)
    row(2, 0, "Silence before sending (s)", ttk.Scale(sp, from_=0.8, to=4.0, variable=v_sil))
    row(2, 1, "Sensitivity", ttk.Scale(sp, from_=0.2, to=0.9, variable=v_sens))
    wf = ttk.Frame(sp)
    ttk.Checkbutton(wf, text="Require wake word:", variable=v_wake).pack(side="left")
    ttk.Entry(wf, textvariable=v["wake_word"], width=14).pack(side="left", padx=6)
    wf.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=3)
    ttk.Button(sp, text="Refresh models", command=lambda: cb_model.config(values=list_models())).grid(row=3, column=2, sticky="w", padx=8)
    ttk.Label(sp, text="System prompt").grid(row=4, column=0, sticky="nw", padx=8, pady=3)
    txt_prompt = tk.Text(sp, height=3, wrap="word", bg=PANEL, fg=FG, insertbackground=FG, bd=0, font=("Segoe UI", 10))
    txt_prompt.insert("1.0", settings["system_prompt"])
    txt_prompt.grid(row=4, column=1, columnspan=3, sticky="ew", padx=8, pady=3)
    sp.columnconfigure(1, weight=1)
    sp.columnconfigure(3, weight=1)

    def settings_changed(*_):
        settings["ollama_model"] = v["ollama_model"].get().strip()
        settings["language"] = v["language"].get().strip() or "en"
        settings["wake_word"] = v["wake_word"].get().strip() or "computer"
        settings["wake_enabled"] = v_wake.get()
        settings["silence_s"] = round(v_sil.get(), 2)
        settings["sensitivity"] = round(v_sens.get(), 2)
        settings["system_prompt"] = txt_prompt.get("1.0", "end").strip() or DEFAULT_PROMPT
        save_settings(settings)

    for var in (v["ollama_model"], v["language"], v["wake_word"], v_wake, v_sil, v_sens):
        var.trace_add("write", settings_changed)
    txt_prompt.bind("<FocusOut>", settings_changed)

    def on_whisper_change(_=None):
        if v["whisper_model"].get() != settings["whisper_model"]:
            settings["whisper_model"] = v["whisper_model"].get()
            save_settings(settings)
            threading.Thread(target=engine.load_whisper, daemon=True).start()

    cb_wh.bind("<<ComboboxSelected>>", on_whisper_change)

    # ----- microphone stream
    stream = {"s": None}

    def start_mic(_=None):
        if stream["s"]:
            stream["s"].close()
            stream["s"] = None
        name = v["mic"].get()
        idx = None
        if name:
            for i, d in enumerate(sd.query_devices()):
                if d["name"] == name and d["max_input_channels"] > 0:
                    idx = i
                    break
        try:
            s = sd.InputStream(device=idx, channels=1, samplerate=SR, blocksize=FRAME, dtype="float32",
                               callback=lambda indata, frames, t, status: engine.feed(indata[:, 0].copy()))
            s.start()
            stream["s"] = s
            used = sd.query_devices(s.device)["name"]
            settings["mic"] = v["mic"].get() or used
            v["mic"].set(settings["mic"])
            save_settings(settings)
            put(f"Microphone: {used}", "sys")
        except Exception as e:  # noqa: BLE001
            put(f"Could not open microphone: {e}", "err")

    cb_mic.bind("<<ComboboxSelected>>", start_mic)

    def toggle_settings():
        settings["show_settings"] = not settings["show_settings"]
        save_settings(settings)
        sp.pack(fill="x", before=bar) if settings["show_settings"] else sp.pack_forget()

    btn_set.config(command=toggle_settings)
    if settings["show_settings"]:
        sp.pack(fill="x", before=bar)

    # ----- engine events -> widgets
    ai = {"open": False}

    def handle(kind, kw):
        if kind == "state":
            text, color = BANNER[kw["name"]]
            banner.config(text=text, bg=color)
            btn_pause.config(text="Resume" if kw["name"] == "paused" else "Pause")
        elif kind == "level":
            level["value"] = kw["p"] * 100
        elif kind == "user":
            put("You:  " + kw["text"], "you")
        elif kind == "ai_start":
            chat.config(state="normal")
            chat.mark_set("aistart", "end-1c")
            chat.mark_gravity("aistart", "left")
            chat.config(state="disabled")
            ai["open"] = True
        elif kind == "ai_set":
            chat.config(state="normal")
            chat.delete("aistart", "end-1c")
            chat.insert("aistart", kw["text"] if kw["text"] or not kw["thinking"] else "(thinking...)", "ai")
            chat.config(state="disabled")
            chat.see("end")
        elif kind == "ai_end":
            if ai["open"]:
                chat.config(state="normal")
                chat.insert("end", "\n", "ai")
                if kw.get("stopped"):
                    chat.insert("end", "(stopped)\n", "sys")
                chat.config(state="disabled")
                ai["open"] = False
        elif kind == "system":
            put(kw["text"], "sys")
        elif kind == "error":
            put(kw["text"], "err")
        elif kind == "ignored":
            put("(ignored: " + kw["text"] + ")", "ign")
        elif kind == "clear":
            clear_chat()
            put("New conversation.", "sys")

    def poll():
        try:
            while True:
                handle(*evq.get_nowait())
        except queue.Empty:
            pass
        root.after(40, poll)

    # ----- go
    models = list_models()
    if not settings["ollama_model"] and models:
        settings["ollama_model"] = pick_default_model(models)
        v["ollama_model"].set(settings["ollama_model"])
        save_settings(settings)
    engine.emit("state", name="loading")
    engine.start()
    start_mic()
    poll()

    def on_close():
        engine.stop()
        if stream["s"]:
            stream["s"].close()
        root.destroy()
        engine.release_models()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
