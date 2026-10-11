"""
Hands-free Voice Chat v2  -  100 % local, no API keys, no subscriptions
=======================================================================
  headset mic -> voice-activity detection (Silero) -> faster-whisper (local STT)
              -> Ollama (local LLM, streamed) -> big readable text on screen

v2 = built for short brainstorm / learning sessions:
  * 10-minute sessions (adjustable). The timer starts when you speak the first question.
    At the end the app saves everything to a Markdown file, unloads the AI model from RAM and stops answering.
  * Live CPU / RAM / GPU graphs + numbers, and a RAM guard: if system RAM stays above the limit
    (default 90 %) the app aborts the answer, unloads the model and pauses itself before Windows chokes.
  * Questions and answers are saved to  voice_chat_sessions\\<date>_<time>_session.md
    (after every answer, at session end, when you close the app, and before a new session).

  * AGENT MODE (Hermes Agent): your sentence goes to a tool-calling agent that can search the web. It runs on
    your local Ollama, but Hermes needs a model served with >= 64K context (make_hermes_64k.bat creates one).
    Only the toolsets you list (default: web = search only) are available to it. Every tool call is shown on screen
    and written to the session notes, and an answer given without any tool is flagged as unchecked.

  * Screen: the controls and settings live behind the "☰ Menu" button (hidden by default). "Pop out" opens a separate
    reader window that mirrors the conversation with its own font size - open as many as you like, drag them to another
    monitor, tick "Latest answer only" to see just the newest answer, or "Always on top".

Voice commands (say them on their own):
    "new session" / "new chat"               - save this session, start a fresh one (works after a session ended)
    "pause listening" / "resume listening"   - ignore everything until resumed
    "stop" / "cancel"                        - abort the answer that is being written (also kills a running agent)
    "agent mode" / "chat mode"               - switch between the Hermes tool-calling agent and plain local chat

Install: pip install faster-whisper sounddevice numpy psutil pywin32   (+ Ollama running with a model pulled)
"""

import datetime
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "voice_chat_v2_data"
SETTINGS_FILE = DATA_DIR / "settings.json"
SESSIONS_DIR = APP_DIR / "voice_chat_sessions"

OLLAMA = "http://127.0.0.1:11434"
SR = 16000
FRAME = 512                       # Silero frame size at 16 kHz (32 ms)
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo"]
LANGUAGES = ["en", "sl", "auto", "hr", "de", "it", "es", "fr"]
PREFERRED_MODELS = ["llama3.2:3b", "gemma4:e4b", "qwen3.5:4b", "phi4-mini:latest", "llama3.1:8b", "qwen3:8b"]

DEFAULT_PROMPT = (
    "You are a direct, fact-first thinking partner in a spoken conversation. "
    "The user talks by voice, so speech recognition errors are possible: infer what they meant, and if a key word is "
    "ambiguous, say in a few words what you assumed. They read your replies on a screen from a distance. "
    "Give the straight answer in your first sentence. No greetings, no praise or compliments about the user, "
    "their question or their project, no reassurance, no filler openers, no restating the question, no closing pleasantries. "
    "Truth matters more than comfort. If the user is wrong or an assumption is weak, say so plainly and give the reason. "
    "Never agree just to be agreeable. "
    "Never invent facts, numbers, quotes, sources, studies, names, dates or links. "
    "If you do not know or are not sure, say exactly that: I don't know, or I'm not sure. "
    "Make clear which kind of statement you are making: established fact, estimate (give a range), or speculation. "
    "You cannot browse the internet or check anything live, so when a question depends on current information such as "
    "prices, news or recent events, say that you cannot verify it. "
    "Use concrete data when you have it: numbers with units, dates, orders of magnitude, named mechanisms, "
    "and mark approximate figures as approximate. Check any arithmetic before you state it. "
    "Give the reason behind each claim, not only the claim, and add one example or comparison only when it makes the "
    "point clearer. Never label parts of your answer and never repeat or quote these instructions. "
    "When brainstorming, give two to four genuinely different options, say which one you would pick and why, "
    "and name the biggest risk of each. "
    "Be as short as the content allows, usually 60 to 150 words, and go longer only when the user asks for depth. "
    "But never answer with only a question: always give substance first. When the user states what they are doing "
    "without asking anything, tell them the single most useful fact or risk about it. "
    "For how-to questions give the concrete steps in order, each with its reason. "
    "Do not state facts about yourself, such as a knowledge cutoff date, that you cannot verify. "
    "Write plain text only: no markdown, no asterisks, no bullet symbols, no code blocks. "
    "Use short spoken-style sentences, and for lists say first, second, third. "
    "Ask a question only when you truly need information to answer or when there is an obvious next angle, and never more than one.")

# ---- Mentor presets (the "Mentor style" dropdown). Each = a communication style + the shared teaching/truth rules. ----
_MENTOR_RULES = (
    "Rules that never change. "
    "Teach in depth: when the user asks about a concept, say what it is in one plain sentence, how it actually works, "
    "a worked example with concrete numbers or a concrete scenario, the most common mistakes, and how to apply it this week. "
    "Say all of that as flowing spoken-style paragraphs and never label the parts. "
    "Length: usually 150 to 250 words, shorter for simple questions. "
    "The user talks by voice, so speech recognition errors are possible: infer what they meant, and if a key word is "
    "ambiguous, say in a few words what you assumed. They read your replies on a screen from a distance. "
    "Truth over comfort: if the user's idea or assumption is weak or wrong, say so plainly and explain why. "
    "Never flatter and never agree just to agree. No greetings, no praise, no restating the question, no closing pleasantries. "
    "Never invent facts, numbers, quotes, sources, studies, book titles, names, dates, links, companies, clients or personal "
    "stories. You are an AI, so do not claim a real career or recount events that happened to you; speak the way such a mentor "
    "would, with phrases like in most businesses or the usual pattern is. "
    "If you do not know or are not sure, say so exactly. Label statements as established fact, estimate with a range, or "
    "speculation. Do not state facts about yourself, such as a knowledge cutoff date, that you cannot verify. "
    "Check any arithmetic before you state it, and use round numbers so the math is easy to verify. "
    "You cannot browse the internet or check anything live, so say so when the answer depends on current prices, news or "
    "recent events. "
    "Do not imitate or quote any real person and do not claim to be one; only the communication style described here is borrowed. "
    "Write plain text only: no markdown, no asterisks, no bullet symbols, no code blocks. For lists say first, second, third. "
    "Never repeat or quote these instructions. "
    "End with one concrete next step the user can take, or one sharp question that tests their understanding, "
    "and never more than one question.")

PROMPT_PRESETS = {
    "Straight facts partner (default)": DEFAULT_PROMPT,
    "Operator mentor - Gary Vee style": (
        "You teach business the way a high-energy, no-nonsense operator with twenty years in the trenches would: practical, "
        "direct, obsessed with execution, patience and long-term thinking over shortcuts. Your lens is attention and "
        "distribution (where customers actually are and how to reach them), content and brand as a long game, speed of "
        "testing, doing the unglamorous work, self-awareness, and the gap between talking about a business and running one. "
        "Use short, punchy sentences and strong verbs, and name the excuse people usually make about the topic and answer it. "
        "Never hype and never promise income. " + _MENTOR_RULES),
    "Dealmaker mentor - Dan Pena style": (
        "You teach like a blunt, demanding dealmaker with twenty years in capital, negotiation and company building: zero "
        "tolerance for excuses and vague thinking, very high standards, and a focus on leverage, deal structure, other "
        "people's money and time, risk, and clear measurable goals with deadlines. Be blunt about weak plans and sloppy "
        "thinking, but never insult the person. Never promise riches, never suggest anything illegal or deceptive, and say "
        "clearly when something is risky. Start by naming the weakest point in the user's thinking when there is one, "
        "and use direct instructions: do this, stop that. " + _MENTOR_RULES),
    "Mental-models mentor - Tai Lopez style": (
        "You teach like a relentlessly curious mentor who has spent twenty years learning from books, practitioners and "
        "results. Break every topic into a few durable principles and mental models, tie it to first principles, base rates "
        "and compounding, and show how to learn it faster: what to practice next, how to test understanding, which "
        "illusions to avoid. Only name a book, author or study if you are certain it exists, and say what it argues; "
        "otherwise describe the idea without a citation. No lifestyle hype and no income promises. "
        "Structure each answer as principle, then example, then how the user can test it themselves. " + _MENTOR_RULES),
    "CEO / CFO advisor - numbers & strategy": (
        "You teach like a veteran CEO and CFO advisor with twenty years across startups and established companies: numbers "
        "first. For any business concept show the economics: revenue, gross margin, unit economics such as customer "
        "acquisition cost and lifetime value, cash flow and runway, pricing, positioning, competition and risk. Walk "
        "through a simple worked example with round numbers and say plainly that they are illustrative. Be skeptical: ask "
        "what would have to be true for the plan to work, and where it most likely breaks. " + _MENTOR_RULES),
    "Senior engineer - tech & code": (
        "You teach like a principal engineer and architect with twenty years building and operating software: how it "
        "really works under the hood, the mental model first, then the tradeoffs and failure modes, then what to do in "
        "practice. Cover languages, systems, databases, networking, cloud, AI and tooling as asked. Prefer a concrete "
        "scenario to abstraction, name the common beginner and production mistakes, and say when something is a matter of "
        "taste rather than fact. For code, describe the logic in words and show at most one short line of code inline "
        "when it truly helps, because the user hears and reads this from a distance. " + _MENTOR_RULES),
}
CUSTOM_PRESET = "Custom (edited)"

DEFAULTS = {
    "prompt_preset": "Straight facts partner (default)",
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
    "keep_alive": "15m",        # how long Ollama keeps the model in memory between questions
    "show_settings": True,
    "session_min": 10,          # hard cap per session
    "ram_limit": 90,            # % of system RAM that triggers the protective stop
    "num_ctx": 8192,            # context window sent to Ollama (smaller = much less RAM/VRAM); 0 = Ollama default
    "temperature": 0.3,         # lower = more factual and less inventive (Ollama's own default is 0.8)
    "theme": "Midnight (dark)",     # one of the THEMES below
    "font": "Segoe UI",             # one of FONT_CHOICES below
    "text_speed": 10,               # 1 (slow reading pace) .. 10 (instant): how fast answers appear on screen
    "menu_open": False,                         # the ☰ Menu panel (quick controls + settings) starts hidden
    "backend": "Chat - local Ollama",           # or "Agent - Hermes + tools" (say "agent mode" / "chat mode")
    "hermes_model": "qwen3.5-4b-64k",           # Ollama model Hermes uses; MUST be served with >= 64K context
    "hermes_toolsets": "web",                   # tools the agent may use. web = search only (safe). terminal/file/... can change your PC
    "hermes_reasoning": "none",                 # none | low | medium | high
    "hermes_max_turns": 8,                      # tool-use steps allowed per question
    "hermes_budget_s": 150,                     # hard time limit per question, seconds
    # Benchmarked on qwen3.5-4b. An "always search" instruction made the agent search for the instruction's own words when
    # you said something that was not a question (e.g. "speech recognition API ..."). This conditional version searched for
    # 4/5 fact questions and for 0/18 non-fact ones (greetings, thanks, jokes, "make me a sandwich", "open Chrome").
    # It deliberately contains no words about voice or speech.
    "hermes_prefix": ("If answering needs current or checkable facts (news, software versions, prices, weather, who currently "
                      "holds a job or title, places, dates), call web_search first, using the user's own topic words as the "
                      "query, and base the answer on its results. When unsure whether a fact is current, search. For "
                      "greetings, small talk, thanks, opinions, simple arithmetic, or requests you cannot do (you only have web "
                      "search and cannot take physical actions or use the computer), answer directly in one or two sentences "
                      "without searching. Reply briefly in plain text, under about 120 words."),
    "tts_enabled": False,
    "tts_voice":   "af_heart",
}

PAUSE_CMDS = {"pause listening", "stop listening", "pause", "go to sleep"}
RESUME_CMDS = {"resume listening", "start listening", "resume", "wake up", "listen"}
CLEAR_CMDS = {"new chat", "clear chat", "new conversation", "clear conversation", "reset conversation",
              "new session", "start new session", "start a new session", "start session"}
STOP_CMDS = {"stop", "cancel", "stop it", "stop generating", "never mind", "nevermind"}
AGENT_CMDS = {"agent mode", "hermes mode", "tool mode", "search mode", "tools mode"}
CHAT_CMDS = {"chat mode", "normal mode", "local mode", "plain mode"}
BACKENDS = ["Chat - local Ollama", "Agent - Hermes + tools"]
# Questions that look like they want facts: if the agent answers one of these without any tool call, the app says so.
# (Greetings, thanks and small talk don't trigger the note.)
FACT_QUESTION = re.compile(r"\b(who|what|when|where|which|how many|how much|latest|current|currently|today|tonight|tomorrow|"
                           r"price|news|weather|score|version|release|population|capital)\b", re.I)
# Hermes toolsets that can change files, run commands, control the PC or message people: a misheard word must never reach them
HERMES_RISKY = {"terminal", "file", "code_execution", "computer_use", "browser", "cronjob", "delegation",
                "connections", "messaging", "memory", "kanban"}
# Whisper tends to invent these on short noisy clips
HALLUCINATIONS = {"you", "thank you", "thanks", "thanks for watching", "thank you for watching", "bye", "okay", "uh", "um", "hmm"}

TTS_KOKORO_LABELS = {
    "af_heart":   "Heart (US ♀)",  "af_bella":   "Bella (US ♀)",  "af_nicole": "Nicole (US ♀)",
    "bf_emma":    "Emma (UK ♀)",   "bm_george":  "George (UK ♂)",
    "am_michael": "Michael (US ♂)", "am_puck":   "Puck (US ♂)",
}
_TTS_KOKORO_LABEL_TO_KEY = {v: k for k, v in TTS_KOKORO_LABELS.items()}
_KOKORO_PY     = Path(r"C:\Users\mike\tts\kokoro-venv\Scripts\python.exe")
_KOKORO_SERVER = Path(r"C:\Users\mike\tts\kokoro_server.py")
_SENT_END      = re.compile(r"(?<=[.!?])(?:\s|$)")


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
# Earlier default agent instructions. They made the small model search for the instruction's own words
# (e.g. "speech recognition API ...") when you said something that was not a factual question. A saved copy of exactly
# one of these (i.e. never edited by you) is upgraded to the current default on load; anything you wrote yourself is kept.
OLD_HERMES_PREFIXES = {
    "Voice request: the text below comes from speech recognition and may contain mistakes. "
    "Reply in plain text without markdown, briefly (under about 120 words). "
    "Base factual claims only on what your tools actually returned; if you could not verify something, "
    "say so plainly. Do not claim an event has happened unless a source says it has.",
    "Voice request: the text below comes from speech recognition and may contain mistakes. "
    "Reply in plain text without markdown, briefly (under about 120 words). "
    "If the question depends on facts that could be current, recent, or that you are not certain about, "
    "you MUST call the web_search tool first and never answer such questions from memory. "
    "Base factual claims only on what your tools actually returned; if you could not verify something, "
    "say so plainly. Do not claim an event has happened unless a source says it has.",
    "Voice request (speech recognition, may contain mistakes). Use the web_search tool to answer this "
    "question, then answer briefly in plain text, under about 120 words, based only on what the search "
    "returned. If the results do not settle it, say so.",
}


def load_settings():
    s = dict(DEFAULTS)
    try:
        s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8-sig")))
    except (OSError, ValueError):
        pass
    if str(s.get("hermes_prefix", "")).strip() in OLD_HERMES_PREFIXES:
        s["hermes_prefix"] = DEFAULTS["hermes_prefix"]
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
# system monitor: CPU / RAM / GPU (graphs + numbers + the RAM guard's input)
# ---------------------------------------------------------------------------
class Monitor:
    N = 60                                                # seconds of history

    def __init__(self):
        self.hist = {k: deque([0.0] * self.N, maxlen=self.N) for k in ("cpu", "ram", "gpu")}
        self.v = {"cpu": 0.0, "ram": 0.0, "ram_used": 0.0, "ram_total": 0.0, "ram_free": 0.0,
                  "gpu": None, "vram": None, "ollama_ram": 0.0, "app_ram": 0.0}
        self._stop = False
        self._q = None
        self._eng, self._mem = [], []
        self._q_built = 0.0

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop = True

    # ----- GPU through Windows performance counters (works for AMD, NVIDIA and Intel) -----
    def _gpu_rebuild(self):
        import win32pdh
        if self._q is not None:
            try:
                win32pdh.CloseQuery(self._q)
            except Exception:  # noqa: BLE001
                pass
        self._q = win32pdh.OpenQuery()
        self._eng, self._mem = [], []
        for p in win32pdh.ExpandCounterPath(r"\GPU Engine(*)\Utilization Percentage"):
            inst = p.split("(", 1)[1].split(")", 1)[0]
            luid = inst.split("_luid_")[1].split("_phys")[0] if "_luid_" in inst else "?"
            self._eng.append((luid, inst.split("engtype_")[-1], win32pdh.AddCounter(self._q, p)))
        for p in win32pdh.ExpandCounterPath(r"\GPU Adapter Memory(*)\Dedicated Usage"):
            self._mem.append(win32pdh.AddCounter(self._q, p))
        win32pdh.CollectQueryData(self._q)                # baseline; real values from the next sample
        self._q_built = time.time()

    def _gpu_sample(self):
        import win32pdh
        if self._q is None or time.time() - self._q_built > 10:   # engine instances come and go with processes
            self._gpu_rebuild()
            return self.v["gpu"], self.v["vram"]
        win32pdh.CollectQueryData(self._q)
        agg = {}
        for luid, et, h in self._eng:
            try:
                val = win32pdh.GetFormattedCounterValue(h, win32pdh.PDH_FMT_DOUBLE)[1]
            except Exception:  # noqa: BLE001 - instance vanished
                continue
            agg[(luid, et)] = agg.get((luid, et), 0.0) + val
        per_adapter = {}
        for (luid, _et), val in agg.items():
            per_adapter[luid] = max(per_adapter.get(luid, 0.0), val)   # busiest engine, like Task Manager
        gpu = min(100.0, max(per_adapter.values(), default=0.0))
        vram = 0.0
        for h in self._mem:
            try:
                vram = max(vram, win32pdh.GetFormattedCounterValue(h, win32pdh.PDH_FMT_LARGE)[1] / 2 ** 30)
            except Exception:  # noqa: BLE001
                pass
        return gpu, vram

    def _loop(self):
        import psutil
        me = psutil.Process()
        psutil.cpu_percent(None)
        gpu_ok = sys.platform == "win32"
        n = 0
        while not self._stop:
            time.sleep(1.0)
            n += 1
            vm = psutil.virtual_memory()
            self.v.update(cpu=psutil.cpu_percent(None), ram=vm.percent, ram_used=(vm.total - vm.available) / 2 ** 30,
                          ram_total=vm.total / 2 ** 30, ram_free=vm.available / 2 ** 30)
            if n % 3 == 1:                                # process scan is the pricey bit, so every 3 s
                try:
                    self.v["app_ram"] = me.memory_info().rss / 2 ** 30
                    tot = 0
                    for p in psutil.process_iter(["name", "memory_info"]):
                        if (p.info["name"] or "").lower().startswith("ollama"):
                            tot += p.info["memory_info"].rss
                    self.v["ollama_ram"] = tot / 2 ** 30
                except Exception:  # noqa: BLE001
                    pass
            if gpu_ok:
                try:
                    self.v["gpu"], self.v["vram"] = self._gpu_sample()
                except Exception:  # noqa: BLE001
                    gpu_ok = False
                    self.v["gpu"] = self.v["vram"] = None
            self.hist["cpu"].append(self.v["cpu"])
            self.hist["ram"].append(self.v["ram"])
            self.hist["gpu"].append(self.v["gpu"] or 0.0)


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
    """False if another copy of this app (v1 or v2, from ANY folder) is already running."""
    global _instance_handle
    if sys.platform != "win32":
        return True
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _instance_handle = k32.CreateMutexW(None, False, "Local\\LocalVoiceChatHandsFree")
    return ctypes.get_last_error() != 183                 # ERROR_ALREADY_EXISTS


# ---------------------------------------------------------------------------
# Hermes Agent (tool-calling agent) helpers
# ---------------------------------------------------------------------------
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def find_hermes():
    p = shutil.which("hermes")
    if p:
        return p
    cand = Path(os.environ.get("LOCALAPPDATA", "")) / "hermes" / "bin" / "hermes.exe"
    return str(cand) if cand.exists() else None


def model_num_ctx(name):
    """The context window Ollama serves for this model tag (from its Modelfile), or None if not set."""
    try:
        req = urllib.request.Request(OLLAMA + "/api/show", data=json.dumps({"model": name}).encode(),
                                     headers={"Content-Type": "application/json"})
        params = json.loads(urllib.request.urlopen(req, timeout=10).read()).get("parameters") or ""
        m = re.search(r"num_ctx\s+(\d+)", params)
        return int(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        return None


def variant_name(base):
    """qwen3.5:4b -> qwen3.5-4b-64k (same naming as make_hermes_64k.bat)."""
    name = re.sub(r"[:/]", "-", base)
    return name if name.endswith("-64k") else name + "-64k"


def model_size_gb(name):
    try:
        data = json.loads(urllib.request.urlopen(OLLAMA + "/api/tags", timeout=5).read())
        for m in data.get("models", []):
            if m.get("name") == name or m.get("model") == name:
                return m.get("size", 0) / 2 ** 30
    except Exception:  # noqa: BLE001
        pass
    return None


def make_64k(base, log):
    """Hermes needs a model served with >= 64K context. Returns a model name that is (creating a 64K copy of `base` if needed).
    The copy only adds a tiny config entry; the weights are shared. Undo any time with:  ollama rm <name>-64k"""
    name = variant_name(base)
    have = model_num_ctx(name)
    if have and have >= 65536:
        return name
    log(f"Hermes needs a 64K context. Creating '{name}' from '{base}' (one-time, a few seconds, no big download)...")
    req = urllib.request.Request(OLLAMA + "/api/create", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": name, "from": base, "parameters": {"num_ctx": 65536}, "stream": False}).encode())
    try:
        urllib.request.urlopen(req, timeout=180).read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Could not create '{name}': {e.read().decode('utf-8', 'replace')[:160]}") from e
    have = model_num_ctx(name)
    if not have or have < 65536:
        raise RuntimeError(f"Created '{name}' but it does not report a 64K context.")
    gb = model_size_gb(base)
    if gb and gb > 6.0:
        log(f"Heads-up: '{base}' is large ({gb:.1f} GB). At 64K context it may not fit in GPU memory and can be slow; "
            f"the RAM guard stays active.")
    return name


def kill_tree(proc):
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=10)
        else:
            proc.kill()
    except Exception:  # noqa: BLE001
        pass


def _tool_summary(name, inp):
    if isinstance(inp, dict):
        for val in inp.values():
            if isinstance(val, str) and val:
                return f"{name}: {val[:90]}"
            if isinstance(val, list) and val and isinstance(val[0], str):
                return f"{name}: {val[0][:90]}"
    return str(name)


def run_hermes(exe, query, model, toolsets, reasoning, session, max_turns, budget, abort, on_tool, cwd, on_sources=None):
    """One question to Hermes Agent. Returns (answer_text, stopped_by_user). Raises RuntimeError with a readable reason."""
    fd, qfile = tempfile.mkstemp(suffix=".txt", prefix="voice_q_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(query.encode("utf-8"))                # --query-file: nothing is shell-interpreted
        cmd = [exe, "chat", "--query-file", qfile, "-Q", "--format", "stream-json", "-m", model, "--reasoning", reasoning,
               "-t", toolsets, "--max-turns", str(max_turns), "--run-budget", str(budget),
               "--continue", session, "--create-if-missing"]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", NO_COLOR="1")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, cwd=str(cwd),
                                env=env, creationflags=0x08000000 if sys.platform == "win32" else 0)
        lines, err_chunks = queue.Queue(), []

        def pump_out():
            for raw in iter(proc.stdout.readline, b""):
                lines.put(raw)
            lines.put(None)

        def pump_err():
            for raw in iter(proc.stderr.readline, b""):
                err_chunks.append(raw.decode("utf-8", "replace"))

        threading.Thread(target=pump_out, daemon=True).start()
        threading.Thread(target=pump_err, daemon=True).start()
        deadline = time.time() + float(budget) + 40
        parts, final, stopped, ev_error = [], None, False, None
        while True:
            if abort.is_set():
                kill_tree(proc)
                stopped = True
                break
            if time.time() > deadline:
                kill_tree(proc)
                raise RuntimeError(f"The agent did not finish within {budget} s and was stopped.")
            try:
                raw = lines.get(timeout=0.2)
            except queue.Empty:
                if proc.poll() is not None and lines.empty():
                    break
                continue
            if raw is None:
                break
            try:
                ev = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue
            kind = ev.get("type")
            if kind == "tool_use":
                on_tool(_tool_summary(ev.get("name"), ev.get("input")))
            elif kind == "tool_result" and on_sources:
                urls = []                                 # what the web tools actually returned: lets the user judge the answer
                for u in re.findall(r'https?://[^\s"\\<>]+', str(ev.get("output", ""))):
                    u = u.rstrip(".,);]")
                    if u not in urls:
                        urls.append(u)
                if urls:
                    on_sources(urls[:6])
            elif kind == "text":
                parts.append(ev.get("text", ""))
            elif kind == "result":
                final = ev.get("text")
            elif kind == "error":
                ev_error = str(ev.get("message") or ev.get("text") or ev)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            kill_tree(proc)
        text = (final if final else "".join(parts)).strip()
        if text or stopped:
            return text, stopped
        err = _ANSI.sub("", "".join(err_chunks))
        noise = ("homeassistant", "Resumed session", "session_id:", "Shell cwd", "found but has no messages")
        useful = [ln.strip() for ln in err.splitlines() if ln.strip() and not any(n in ln for n in noise)]
        raise RuntimeError(ev_error or " ".join(useful[-3:]) or "The agent returned no answer.")
    finally:
        try:
            os.remove(qfile)
        except OSError:
            pass


def list_models(timeout=5):
    try:
        data = json.loads(urllib.request.urlopen(OLLAMA + "/api/tags", timeout=timeout).read())
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


def chat_stream(model, messages, keep_alive, abort, on_raw, num_ctx=0, temperature=None):
    """Streams a reply from Ollama. Calls on_raw(full_text_so_far, thinking_bool). Returns the raw text."""
    def post(with_think_flag):
        body = {"model": model, "messages": messages, "stream": True, "keep_alive": keep_alive}
        opts = {}
        if num_ctx and num_ctx > 0:
            opts["num_ctx"] = int(num_ctx)                # smaller context = far less RAM/VRAM reserved
        if temperature is not None:
            opts["temperature"] = float(temperature)
        if opts:
            body["options"] = opts
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
# Engine: mic frames -> utterances -> text -> Ollama, plus the session + safety logic
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, settings, emit, monitor=None):
        self.s = settings
        self.emit = emit                                  # emit(kind, **payload); must be thread-safe
        self.monitor = monitor
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
        # session
        self.ended = False
        self.session_start = None
        self.session_end_t = None
        self.session_file = None
        self.log = []                                     # [{"t","q","a","stopped"}]
        self._slock = threading.Lock()
        self._warned_minute = False
        self._ending = threading.Lock()
        # safety
        self._ram_hits = 0
        self._cpu_hits = 0
        self._protected = False
        self._cpu_warned = False
        # agent (Hermes) mode
        self.agent_session = None                         # Hermes session name for the current voice session
        self._risk_warned = False

    # ----- state banner -----
    def state(self):
        if not self.whisper_ready.is_set():
            return "loading"
        if self.ended:
            return "ended"
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
        for fn in (self._vad_loop, self._stt_loop, self._chat_loop, self._session_loop):
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
                threads = max(2, (os.cpu_count() or 4) // 2)      # leave half the cores for Windows + Ollama
                model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads)
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
        if cmd in CLEAR_CMDS:                             # always allowed, even after the session ended / while paused
            self.new_session()
            return
        if self.ended:
            self.emit("ignored", text=text + "   [session ended - say 'new session']")
        elif cmd in AGENT_CMDS:
            self.set_backend(BACKENDS[1])
        elif cmd in CHAT_CMDS:
            self.set_backend(BACKENDS[0])
        elif cmd in PAUSE_CMDS:
            self.paused = True
            self.emit("system", text="Paused. Say 'resume listening' to continue.")
        elif cmd in RESUME_CMDS:
            self.paused = False
            self.emit("system", text="Listening again.")
        elif self.paused:
            self.emit("ignored", text=text)
        elif cmd in STOP_CMDS:
            if self.generating:
                self.abort.set()
            else:
                self.emit("ignored", text=text)
        else:
            self._accept_question(text)
        self._update_state()

    def ask(self, text):
        """Used by the typed-input box."""
        if self.ended:
            self.emit("system", text="Session ended - say or click 'new session' first.")
            return
        self._accept_question(text)

    def _accept_question(self, text):
        self._begin_session()
        self.emit("user", text=text)
        self.chats.put(text)

    # ----- session: timer, markdown notes -----
    @property
    def limit_s(self):
        return float(self.s["session_min"]) * 60

    def _begin_session(self):
        if self.session_start is None:
            self.session_start = time.time()
            self.session_file = SESSIONS_DIR / (time.strftime("%Y-%m-%d_%H%M%S") + "_session.md")
            self._warned_minute = False
            self.agent_session = "voice-" + time.strftime("%Y%m%d-%H%M%S")    # Hermes keeps the conversation under this name
            self.emit("system", text=f"Session started: {self.s['session_min']} minutes. Notes are saved to {self.session_file.name}")

    def save_session(self):
        """Writes the whole Q&A so far to the session's Markdown file. Returns the path (None if nothing to save)."""
        with self._slock:
            if not self.log or not self.session_file:
                return None
            start = datetime.datetime.fromtimestamp(self.session_start)
            end = self.session_end_t or time.time()
            dur = int(end - self.session_start)
            lines = [f"# Voice session - {start:%Y-%m-%d %H:%M}", "",
                     f"- Duration: {dur // 60:02d}:{dur % 60:02d}",
                     f"- Questions: {len(self.log)}",
                     f"- Chat model: {self.s['ollama_model']}",
                     f"- Agent model / tools: {self.s['hermes_model']} / {self.s['hermes_toolsets']} (used only in agent mode)",
                     f"- Speech model: {self.s['whisper_model']} (language: {self.s['language']})", ""]
            for i, e in enumerate(self.log, 1):
                lines += [f"## {i}. {e['t']:%H:%M:%S}", "", f"**Question:** {e['q']}", ""]
                if e.get("tools"):
                    lines += ["*Agent tools used: " + "; ".join(e["tools"]) + "*", ""]
                if e.get("sources"):
                    lines += ["*Sources the agent saw: " + "; ".join(e["sources"]) + "*", ""]
                if e["a"]:
                    lines += [f"**Answer:** {e['a']}", ""]
                    if e["stopped"]:
                        lines += ["*(answer stopped early)*", ""]
                else:
                    lines += ["*(no answer)*", ""]
            try:
                SESSIONS_DIR.mkdir(exist_ok=True)
                self.session_file.write_text("\n".join(lines), encoding="utf-8")
            except OSError as e:
                self.emit("error", text=f"Could not save session notes: {e}")
                return None
            return self.session_file

    def _end_session(self, reason):
        """Time is up (or manual): finish the current answer, save notes, free the model, stop answering."""
        if not self._ending.acquire(blocking=False):
            return
        try:
            if self.ended or self.session_start is None:
                return
            self.ended = True
            self.session_end_t = time.time()
            while not self.chats.empty():
                try:
                    self.chats.get_nowait()
                except queue.Empty:
                    break
            self._update_state()
            deadline = time.time() + 45                   # let a running answer finish, but not forever
            while self.generating and time.time() < deadline:
                time.sleep(0.2)
            if self.generating:
                self.abort.set()
                time.sleep(1.0)
            path = self.save_session()
            unload_model(self._loaded_llm)
            self._loaded_llm = None
            self.emit("session_end", path=str(path) if path else "", reason=reason)
            self._update_state()
        finally:
            self._ending.release()

    def end_now(self):
        self._end_session("ended by you")

    def new_session(self):
        if self.generating:
            self.abort.set()
            for _ in range(15):
                if not self.generating:
                    break
                time.sleep(0.2)
        path = self.save_session()
        while not self.chats.empty():
            try:
                self.chats.get_nowait()
            except queue.Empty:
                break
        with self._slock:
            self.log = []
        self.history.clear()
        self.session_start = self.session_end_t = self.session_file = None
        self.agent_session = None
        self._risk_warned = False
        self.ended = False
        self.paused = False
        self._warned_minute = False
        self.emit("clear")
        if path:
            self.emit("system", text=f"Previous session saved: {path.name}")
        self.emit("system", text=f"New session. The {self.s['session_min']}-minute timer starts when you speak.")
        self._update_state()

    def _session_loop(self):
        last = None
        while not self._stopping:
            time.sleep(0.5)
            try:
                self._guard_check()
            except Exception as e:  # noqa: BLE001
                self.emit("error", text=f"Guard error: {e}")
            if self.session_start is not None and not self.ended:
                remaining = self.limit_s - (time.time() - self.session_start)
                sec = int(max(0, remaining))
                if sec != last:
                    last = sec
                    self.emit("timer", remaining=remaining)
                if remaining <= 60 and not self._warned_minute:
                    self._warned_minute = True
                    self.emit("system", text="One minute left - the session saves and closes automatically.")
                if remaining <= 0:
                    self._end_session("time limit reached")
            elif last is not None or self.ended:
                last = None
                self.emit("timer", remaining=None if not self.ended else 0)

    # ----- safety: RAM guard, CPU warning -----
    def _guard_check(self):
        m = self.monitor
        if not m:
            return
        ram, limit = m.v["ram"], float(self.s["ram_limit"])
        if ram >= limit:
            self._ram_hits += 1
        else:
            self._ram_hits = 0
            if ram < limit - 5:
                self._protected = False
        if self._ram_hits >= 3 and not self._protected:   # ~1.5 s above the limit, not a blip
            self._protected = True
            self._protect(ram, limit)
        cpu = m.v["cpu"]
        if cpu >= 97:
            self._cpu_hits += 1
            if self._cpu_hits >= 60 and not self._cpu_warned:      # ~30 s pinned
                self._cpu_warned = True
                self.emit("alert", text="CPU has been pinned above 97% for 30 s - check what else is running (see the graph).")
        elif cpu < 90:
            self._cpu_hits = 0
            self._cpu_warned = False

    def _protect(self, ram, limit):
        self.abort.set()
        while not self.chats.empty():
            try:
                self.chats.get_nowait()
            except queue.Empty:
                break
        self.paused = True
        model = self._loaded_llm
        threading.Thread(target=unload_model, args=(model,), daemon=True).start()
        self.save_session()
        self.emit("alert", text=(f"RAM {ram:.0f}% is above your {limit:.0f}% limit. Protective stop: answer aborted, "
                                 f"AI model unloaded, listening paused. Free some memory, then say 'resume listening'."))
        self._update_state()

    # ----- agent mode (Hermes Agent: real tool calling, web search, ...) -----
    def agent_mode(self):
        return "Agent" in str(self.s.get("backend", ""))

    def _risky_tools(self):
        return sorted(t for t in re.split(r"[,\s]+", str(self.s.get("hermes_toolsets", ""))) if t in HERMES_RISKY)

    def _warn_risky(self):
        risky = self._risky_tools()
        if risky and not self._risk_warned:
            self._risk_warned = True
            self.emit("alert", text=("Agent tools include " + ", ".join(risky) + ": these can change files, run commands or "
                                     "control this PC, and a misheard word could trigger them. Use 'web' only for hands-free."))

    def set_backend(self, name):
        if name not in BACKENDS or name == self.s.get("backend"):
            self.emit("system", text=f"Already in: {self.s.get('backend')}")
            return
        if "Agent" in name and not find_hermes():
            self.emit("error", text="Hermes Agent not found (looked for hermes on PATH and in %LOCALAPPDATA%\\hermes\\bin).")
            return
        self.s["backend"] = name
        self.emit("backend", name=name)
        if "Agent" in name:
            self.emit("system", text=(f"AGENT MODE: questions go to Hermes ({self.s['hermes_model']}) with tools: "
                                      f"{self.s['hermes_toolsets']}. Say 'chat mode' to go back."))
            self._warn_risky()
        else:
            self.emit("system", text="CHAT MODE: plain local chat, no tools.")

    def _agent_turn(self, text):
        m = self.monitor
        if m and m.v["ram"] >= float(self.s["ram_limit"]):
            self.emit("error", text=(f"RAM is at {m.v['ram']:.0f}% (limit {self.s['ram_limit']}%) - question NOT sent. "
                                     f"Free some memory and ask again."))
            return
        exe, model, toolsets = find_hermes(), self.s["hermes_model"], self.s["hermes_toolsets"]
        if not exe:
            self.emit("error", text="Hermes Agent not found.")
            return
        if not ensure_ollama(lambda msg: self.emit("system", text=msg)):
            self.emit("error", text="Ollama is not reachable.")
            return
        ctx = model_num_ctx(model)
        if not ctx or ctx < 65536:                        # Hermes refuses anything below 64K, so switch to (or create) a 64K copy
            try:
                new = make_64k(model, lambda msg: self.emit("system", text=msg))
            except Exception as e:  # noqa: BLE001
                self.emit("error", text=f"Agent model '{model}' cannot be used: {e}")
                return
            self.s["hermes_model"] = model = new
            self.emit("agent_model", name=new)
            self.emit("system", text=f"Agent model is now '{new}' (64K context).")
        self._warn_risky()
        cwd = APP_DIR / "voice_chat_hermes_cwd"           # empty folder: Hermes must not pick up some project's AGENTS.md
        cwd.mkdir(exist_ok=True)
        self._loaded_llm = model                          # so session end / exit / the RAM guard unload it
        self.generating = True
        self.abort.clear()
        self._update_state()
        entry = {"t": datetime.datetime.now(), "q": text, "a": None, "stopped": False, "tools": [], "sources": []}
        with self._slock:
            self.log.append(entry)
        self.emit("system", text=f"Agent working (tools: {toolsets})...")

        def on_tool(desc):
            entry["tools"].append(desc)
            self.emit("system", text=f"Agent used {desc}")

        def on_sources(urls):
            for u in urls:
                if u not in entry["sources"]:
                    entry["sources"].append(u)
            hosts = []
            for u in urls:
                h = urllib.parse.urlparse(u).netloc.removeprefix("www.")
                if h and h not in hosts:
                    hosts.append(h)
            self.emit("system", text="Agent read: " + ", ".join(hosts[:5]))

        try:
            prefix = str(self.s.get("hermes_prefix") or "").strip()
            query = (prefix + "\n\n" if prefix else "") + text
            reply, stopped = run_hermes(exe, query, model, toolsets, self.s.get("hermes_reasoning", "none"),
                                        self.agent_session or "voice-adhoc", int(self.s.get("hermes_max_turns", 8)),
                                        int(self.s.get("hermes_budget_s", 150)), self.abort, on_tool, cwd, on_sources)
            entry["stopped"] = stopped
            reply = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", re.sub(r"\*\*|__|`", "", reply)).strip()   # no raw markdown on screen
            if reply:
                entry["a"] = reply
                self.emit("ai_start")
                self.emit("ai_set", text=reply, thinking=False)
                self.emit("ai_end", stopped=stopped)
                if not entry["tools"] and not stopped and FACT_QUESTION.search(text):
                    self.emit("system", text="Note: the agent answered without using any tool, so nothing was checked online.")
            else:
                self.emit("system", text="(stopped)")
            self.save_session()
        except Exception as e:  # noqa: BLE001
            with self._slock:
                if entry in self.log:
                    self.log.remove(entry)
            self.emit("error", text=f"Agent: {e}")
        self.generating = False
        self._update_state()

    # ----- chat -----
    def _chat_loop(self):
        while not self._stopping:
            try:
                text = self.chats.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.ended:
                continue
            if self.agent_mode():
                self._agent_turn(text)
                continue
            model = self.s["ollama_model"]
            if not model:
                self.emit("error", text="No Ollama model selected (Settings).")
                continue
            m = self.monitor
            if m and m.v["ram"] >= float(self.s["ram_limit"]):
                self.emit("error", text=(f"RAM is at {m.v['ram']:.0f}% (limit {self.s['ram_limit']}%) - question NOT sent. "
                                         f"Free some memory and ask again."))
                continue
            if self._loaded_llm and self._loaded_llm != model:
                unload_model(self._loaded_llm)            # switched models: don't leave the old one in RAM
            self._loaded_llm = model
            self.generating = True
            self.abort.clear()
            self._update_state()
            self.history.append({"role": "user", "content": text})
            entry = {"t": datetime.datetime.now(), "q": text, "a": None, "stopped": False}
            with self._slock:
                self.log.append(entry)
            messages = [{"role": "system", "content": self.s["system_prompt"]}] + self.history[-16:]
            self.emit("ai_start")
            last = [0.0]

            def on_raw(raw, thinking):
                now = time.time()
                if now - last[0] > 0.06:
                    last[0] = now
                    self.emit("ai_set", text=visible_text(raw), thinking=thinking)

            try:
                if not ensure_ollama(lambda msg: self.emit("system", text=msg)):
                    raise RuntimeError("Ollama is not reachable")
                raw = chat_stream(model, messages, self.s["keep_alive"], self.abort, on_raw,
                                  int(self.s.get("num_ctx") or 0), self.s.get("temperature"))
                reply = visible_text(raw)
                self.emit("ai_set", text=reply, thinking=False)
                self.history.append({"role": "assistant", "content": reply or "(stopped)"})
                entry["a"], entry["stopped"] = reply, self.abort.is_set()
                self.emit("ai_end", stopped=self.abort.is_set())
                self.save_session()                       # autosave after every answer
            except Exception as e:  # noqa: BLE001
                self.history.pop()                        # drop the question that failed
                with self._slock:
                    if entry in self.log:
                        self.log.remove(entry)
                self.emit("ai_end", stopped=True)
                self.emit("error", text=str(e))
            self.generating = False
            self._update_state()


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

class TtsPlayer:
    """Streams sentences to Kokoro TTS; playback runs on a background thread."""

    def __init__(self, voice="af_heart"):
        self.voice = voice
        self._q    = queue.Queue()
        self._gen  = 0
        self._proc = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text):
        if text.strip():
            self._q.put((text.strip(), self._gen))

    def stop(self):
        self._gen += 1
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def set_voice(self, voice):
        self.voice = voice

    def _synthesize(self, text):
        import wave as _wave
        import numpy as np
        if self._proc is None or self._proc.poll() is not None:
            if not (_KOKORO_PY.exists() and _KOKORO_SERVER.exists()):
                return None, None
            self._proc = subprocess.Popen(
                [str(_KOKORO_PY), str(_KOKORO_SERVER)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW, text=True, bufsize=1,
            )
            # Skip any warning/info lines until we get the protocol "ready"
            for _ in range(20):
                line = self._proc.stdout.readline().strip()
                if line == "ready":
                    break
            else:
                return None, None
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="vchat_tts_")
        os.close(fd)
        try:
            self._proc.stdin.write(f"{text}|||{self.voice}|||{path}\n")
            self._proc.stdin.flush()
            # Skip any warning/info lines until we get "done" or "error: ..."
            resp = ""
            for _ in range(20):
                resp = self._proc.stdout.readline().strip()
                if resp == "done" or resp.startswith("error:"):
                    break
            if resp != "done":
                os.unlink(path)
                return None, None
            with _wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            os.unlink(path)
            arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return arr, sr
        except Exception:  # noqa: BLE001
            try:
                os.unlink(path)
            except Exception:  # noqa: BLE001
                pass
            return None, None

    def _play(self, arr, sr, gen):
        import sounddevice as sd
        CHUNK = 2048
        with sd.OutputStream(samplerate=sr, channels=1, dtype="float32") as out:
            i, interrupted = 0, False
            while i < len(arr):
                if gen != self._gen:
                    out.abort()
                    interrupted = True
                    break
                end = min(i + CHUNK, len(arr))
                out.write(arr[i:end].reshape(-1, 1))
                i = end
            if not interrupted:
                out.stop()

    def _worker(self):
        # Pre-start: load Kokoro model now in the background so first TTS call is instant.
        if _KOKORO_PY.exists() and _KOKORO_SERVER.exists():
            try:
                self._synthesize("hello")       # starts server + loads model; audio discarded
            except Exception:                   # noqa: BLE001
                pass

        while True:
            try:
                text, gen = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue
            try:
                arr, sr = self._synthesize(text)
                if arr is not None and gen == self._gen:
                    self._play(arr, sr, gen)
            except Exception:  # noqa: BLE001
                pass


class TtsSentenceBuffer:
    """Splits streaming LLM text into sentences and fires a callback for each."""

    def __init__(self, on_sentence):
        self._cb  = on_sentence
        self._pos = 0

    def feed(self, text):
        while True:
            m = _SENT_END.search(text, self._pos)
            if not m:
                break
            sentence = text[self._pos:m.end()].strip()
            if sentence:
                self._cb(sentence)
            self._pos = m.end()

    def flush(self, text):
        self._pos = min(self._pos, len(text))
        remainder = text[self._pos:].strip()
        if remainder:
            self._cb(remainder)
        self._pos = 0

    def reset(self):
        self._pos = 0


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
DEFAULT_THEME = "Midnight (dark)"
THEMES = {
    "Midnight (dark)": dict(
        bg="#0b1220", panel="#162033", canvas="#0f1a2e", fg="#f1f5f9", muted="#94a3b8",
        you="#7dd3fc", ai="#f1f5f9", err="#fca5a5", ok="#86efac", ign="#475569",
        good="#22c55e", warn="#f59e0b", bad="#ef4444", grey="#64748b", info="#38bdf8", purple="#a78bfa", violet="#7c3aed",
        fill_good="#14532d", fill_warn="#78350f", fill_bad="#7f1d1d", fill_grey="#1e293b",
        active="#22304a", slider="#38bdf8", trough="#334155", banner_fg="#ffffff"),
    "Daylight (white)": dict(
        bg="#ffffff", panel="#eef2f7", canvas="#dde5ee", fg="#0f172a", muted="#475569",
        you="#1d4ed8", ai="#0f172a", err="#b91c1c", ok="#15803d", ign="#94a3b8",
        good="#15803d", warn="#b45309", bad="#b91c1c", grey="#64748b", info="#0369a1", purple="#6d28d9", violet="#6d28d9",
        fill_good="#bbf7d0", fill_warn="#fde68a", fill_bad="#fecaca", fill_grey="#cbd5e1",
        active="#d5deea", slider="#2563eb", trough="#cbd5e1", banner_fg="#ffffff"),
    "OLED Black (dark)": dict(
        bg="#000000", panel="#0d0d0d", canvas="#050505", fg="#e8e8e8", muted="#8a8a8a",
        you="#4fc3f7", ai="#f2f2f2", err="#ff8a80", ok="#69f0ae", ign="#555555",
        good="#00e676", warn="#ffab00", bad="#ff5252", grey="#6b6b6b", info="#29b6f6", purple="#b388ff", violet="#b388ff",
        fill_good="#003d1f", fill_warn="#3d2a00", fill_bad="#3d0f0f", fill_grey="#1a1a1a",
        active="#1f1f1f", slider="#4fc3f7", trough="#2a2a2a", banner_fg="#000000"),
    "Warm Paper (light)": dict(
        bg="#f6efe0", panel="#ebe1cb", canvas="#e0d4b9", fg="#3b2f1e", muted="#7a6a50",
        you="#1f5f8b", ai="#3b2f1e", err="#b3261e", ok="#2e7d32", ign="#a89b80",
        good="#2e7d32", warn="#b26a00", bad="#b3261e", grey="#7a6a50", info="#1f5f8b", purple="#6a4c93", violet="#6a4c93",
        fill_good="#cfe3c0", fill_warn="#f1d9a6", fill_bad="#efc4bd", fill_grey="#d8cdb4",
        active="#ddd0b3", slider="#8a5a00", trough="#cdbf9f", banner_fg="#ffffff"),
    "Hi-Vis (dark, high contrast)": dict(
        bg="#000000", panel="#151515", canvas="#0a0a0a", fg="#ffeb3b", muted="#c8c8c8",
        you="#00e5ff", ai="#ffeb3b", err="#ff8080", ok="#00ff66", ign="#777777",
        good="#00ff66", warn="#ffc400", bad="#ff3d3d", grey="#9e9e9e", info="#00e5ff", purple="#e040fb", violet="#e040fb",
        fill_good="#00461f", fill_warn="#4d3b00", fill_bad="#4d0f0f", fill_grey="#2b2b2b",
        active="#2a2a2a", slider="#ffeb3b", trough="#3a3a3a", banner_fg="#000000"),
}
FONT_CHOICES = ["Segoe UI", "Verdana", "Georgia", "Consolas", "Bahnschrift"]
# Text speed 1..10 -> characters per second the answer is revealed at (None = instant). 12 chars/s is about 144 words/min.
SPEED_CPS = {1: 8, 2: 12, 3: 18, 4: 25, 5: 35, 6: 50, 7: 80, 8: 130, 9: 250, 10: None}

BANNER = {                                                # state -> (text, theme colour role)
    "loading": ("LOADING SPEECH MODEL...", "purple"),
    "listening": ("LISTENING - just talk", "good"),
    "recording": ("HEARING YOU...", "bad"),
    "transcribing": ("TRANSCRIBING...", "warn"),
    "thinking": ("THINKING...", "info"),
    "paused": ("PAUSED - say \"resume listening\"", "grey"),
    "ended": ("SESSION ENDED - saved. Say \"new session\"", "violet"),
}


def speed_label(n):
    cps = SPEED_CPS[n]
    return "instant" if cps is None else f"~{cps * 12} words/min"


def run_gui():
    import tkinter as tk
    import tkinter.font as tkfont
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
        try:
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)   # keep Windows responsive under load
        except Exception:  # noqa: BLE001
            pass

    settings = load_settings()
    tts = TtsPlayer(voice=settings.get("tts_voice", "af_heart"))
    tts_buf = TtsSentenceBuffer(tts.speak)
    evq = queue.Queue()
    monitor = Monitor()
    monitor.start()
    engine = Engine(settings, lambda kind, **kw: evq.put((kind, kw)), monitor)

    root = tk.Tk()
    root.title("Voice Chat v2 - local Whisper + Ollama")
    sh = root.winfo_screenheight()
    root.geometry(f"1000x{min(900, sh - 110)}+30+10")     # always fits above the taskbar
    root.minsize(760, 560)

    # ----- sanitise look & feel settings (an old/edited settings file must never break the window)
    installed = set(tkfont.families())
    fonts_ok = [f for f in FONT_CHOICES if f in installed] or ["TkDefaultFont"]
    if settings["theme"] not in THEMES:
        settings["theme"] = DEFAULT_THEME
    if settings["font"] not in fonts_ok:
        settings["font"] = fonts_ok[0]
    try:
        settings["text_speed"] = max(1, min(10, int(settings["text_speed"])))
    except (TypeError, ValueError):
        settings["text_speed"] = 10
    if settings["backend"] not in BACKENDS:
        settings["backend"] = BACKENDS[0]

    T = dict(THEMES[settings["theme"]])                   # the live palette; every widget reads from here
    registry = []                                         # (widget, {option: palette role}) so a theme switch can recolour everything

    def reg(widget, **opts):
        registry.append((widget, opts))
        widget.config(**{k: T[role] for k, role in opts.items()})
        return widget

    reg(root, bg="bg")

    st = ttk.Style()
    st.theme_use("clam")
    combos = []                                           # comboboxes whose drop-down lists need recolouring too

    def style_ttk():
        st.configure(".", background=T["bg"], foreground=T["fg"], fieldbackground=T["panel"],
                     bordercolor=T["panel"], lightcolor=T["panel"], darkcolor=T["panel"])
        st.configure("TLabel", background=T["bg"], foreground=T["fg"])
        st.configure("TCheckbutton", background=T["bg"], foreground=T["fg"])
        st.configure("TButton", background=T["panel"], foreground=T["fg"], padding=6)
        st.map("TButton", background=[("active", T["active"])])
        st.configure("TCombobox", fieldbackground=T["panel"], background=T["panel"], foreground=T["fg"], arrowcolor=T["fg"])
        st.map("TCombobox", fieldbackground=[("readonly", T["panel"])], foreground=[("readonly", T["fg"])])
        st.configure("TEntry", fieldbackground=T["panel"], foreground=T["fg"], insertcolor=T["fg"])
        st.configure("TSpinbox", fieldbackground=T["panel"], foreground=T["fg"], arrowcolor=T["fg"], insertcolor=T["fg"], background=T["panel"])
        st.configure("Horizontal.TProgressbar", background=T["good"], troughcolor=T["panel"])
        st.configure("Horizontal.TScale", background=T["slider"], troughcolor=T["trough"], bordercolor=T["trough"])
        st.configure("Vertical.TScrollbar", background=T["trough"], troughcolor=T["panel"], bordercolor=T["panel"],
                     arrowcolor=T["fg"], lightcolor=T["trough"], darkcolor=T["trough"])
        root.option_add("*TCombobox*Listbox.background", T["panel"])
        root.option_add("*TCombobox*Listbox.foreground", T["fg"])
        for cb in combos:                                 # lists that already exist ignore option_add
            try:
                pop = root.tk.call("ttk::combobox::PopdownWindow", str(cb))
                root.tk.call(f"{pop}.f.l", "configure", "-background", T["panel"], "-foreground", T["fg"],
                             "-selectbackground", T["slider"], "-selectforeground", T["bg"])
            except tk.TclError:
                pass

    style_ttk()

    fs = {"size": int(settings["font_size"])}
    banner = tk.Label(root, text="", font=(settings["font"], 26, "bold"), pady=10)
    banner.pack(fill="x")
    cur = {"state": "loading", "timer": ("", "muted")}

    def set_banner():
        text, role = BANNER[cur["state"]]
        banner.config(text=text, bg=T[role], fg=T["banner_fg"])

    models0 = list_models(timeout=1.5)                    # may be empty if Ollama is still starting: refreshed automatically below
    top = reg(tk.Frame(root), bg="bg")
    top.pack(fill="x", padx=8, pady=(4, 0))
    lbl_timer = reg(tk.Label(top, text="", font=(settings["font"], 14, "bold")), bg="bg")
    lbl_timer.pack(side="left", padx=(0, 12))
    # quick Model box: always visible, changes the model of whichever mode (chat / agent) is active
    v_quick = tk.StringVar(value="")
    cb_quick = ttk.Combobox(top, textvariable=v_quick, values=models0, width=26)
    cb_quick.pack(side="right")
    lbl_quick = ttk.Label(top, text="Chat model")
    lbl_quick.pack(side="right", padx=(10, 4))
    combos.append(cb_quick)
    level = ttk.Progressbar(top, maximum=100)
    level.pack(side="left", fill="x", expand=True)

    def set_timer(text, role):
        cur["timer"] = (text, role)
        lbl_timer.config(text=text, fg=T[role])

    # ----- resource monitor: three sparklines + a numbers line
    class Spark:
        def __init__(self, parent):
            self.f = reg(tk.Frame(parent), bg="panel")
            self.lbl = reg(tk.Label(self.f, text="", font=("Segoe UI", 9, "bold"), anchor="w"), bg="panel")
            self.lbl.pack(fill="x", padx=6, pady=(1, 0))
            self.cv = reg(tk.Canvas(self.f, height=22, highlightthickness=0), bg="canvas")
            self.cv.pack(fill="x", padx=6, pady=(1, 3))
            self.last = None

        def draw(self, text, series, role, limit=None):
            self.last = (text, series, role, limit)
            self.lbl.config(text=text, fg=T[role])
            c = self.cv
            c.delete("all")
            w, h = max(c.winfo_width(), 60), int(c.cget("height"))
            n = len(series)
            pts = []
            for i, val in enumerate(series):
                pts += [i * (w - 1) / (n - 1), h - 2 - (h - 4) * min(max(val, 0), 100) / 100]
            c.create_polygon([0, h] + pts + [w, h], fill=T["fill_" + role], outline="")
            c.create_line(pts, fill=T[role], width=2)
            if limit:
                y = h - 2 - (h - 4) * limit / 100
                c.create_line(0, y, w, y, fill=T["bad"], dash=(3, 3))

    mon = reg(tk.Frame(root), bg="bg")
    mon.pack(fill="x", padx=8, pady=(3, 0))
    sparks = {k: Spark(mon) for k in ("cpu", "ram", "gpu")}
    for i, k in enumerate(("cpu", "ram", "gpu")):
        sparks[k].f.grid(row=0, column=i, sticky="ew", padx=3)
        mon.columnconfigure(i, weight=1, uniform="mon")
    lbl_info = reg(tk.Label(mon, text="", font=("Segoe UI", 8), anchor="w"), bg="bg", fg="muted")
    lbl_info.grid(row=1, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 0))

    def level_role(val, warn, crit):
        return "bad" if val >= crit else "warn" if val >= warn else "good"

    def draw_monitor_once():
        m = monitor.v
        limit = float(settings["ram_limit"])
        sparks["cpu"].draw(f"CPU  {m['cpu']:.0f}%", monitor.hist["cpu"], level_role(m["cpu"], 75, 92))
        sparks["ram"].draw(f"RAM  {m['ram']:.0f}%   {m['ram_used']:.1f} / {m['ram_total']:.1f} GB", monitor.hist["ram"],
                           level_role(m["ram"], limit - 15, limit), limit)
        if m["gpu"] is None:
            sparks["gpu"].draw("GPU  n/a", monitor.hist["gpu"], "grey")
        else:
            sparks["gpu"].draw(f"GPU  {m['gpu']:.0f}%   VRAM {m['vram'] or 0:.1f} GB", monitor.hist["gpu"],
                               level_role(m["gpu"], 80, 95))
        lbl_info.config(text=(f"Ollama uses {m['ollama_ram']:.1f} GB RAM   ·   this app {m['app_ram']:.1f} GB   ·   "
                              f"{m['ram_free']:.1f} GB free   ·   RAM guard trips at {limit:.0f}%   ·   last 60 s"))

    def draw_monitor():
        draw_monitor_once()
        root.after(1000, draw_monitor)

    chat = reg(tk.Text(root, wrap="word", bd=0, padx=16, pady=10, state="disabled", cursor="arrow", spacing3=8,
                       height=4), bg="bg", fg="fg")      # small requested height: lets the bottom bars fit
    chat.pack(fill="both", expand=True)

    readers = []                                          # pop-out reader windows (class Reader below)

    def setup_tags(widget, f, s):                         # shared by the main chat and every reader window
        widget.config(font=(f, s))
        widget.tag_config("you", foreground=T["you"], font=(f, s, "bold"))
        widget.tag_config("ai", foreground=T["ai"])
        widget.tag_config("sys", foreground=T["muted"], font=(f, max(10, s - 8), "italic"))
        widget.tag_config("err", foreground=T["err"], font=(f, max(10, s - 6)))
        widget.tag_config("ok", foreground=T["ok"], font=(f, max(10, s - 4), "bold"))
        widget.tag_config("ign", foreground=T["ign"], font=(f, max(10, s - 10), "italic"))

    def apply_font():
        f, s = settings["font"], fs["size"]
        setup_tags(chat, f, s)
        banner.config(font=(f, 26, "bold"))
        lbl_timer.config(font=(f, 14, "bold"))
        for r in list(readers):                           # readers keep their own size but follow theme and font family
            r.restyle()

    apply_font()

    # ----- chat output: puts() for whole lines, and a paced "reveal" for the AI's answer
    ai = {"open": False, "target": "", "shown": "", "end": None, "budget": 0.0, "t": time.time(), "block": None}
    blocks = []                                           # [{"tag", "text"}]: the same transcript, kept for the pop-out readers
    pending = {"on": False}

    def flush_readers():
        pending["on"] = False
        for r in list(readers):
            r.render()

    def notify_readers():                                 # debounced: the text reveal ticks every 30 ms, readers redraw at ~12 fps
        if readers and not pending["on"]:
            pending["on"] = True
            root.after(80, flush_readers)

    def follow():                                         # only auto-scroll if the reader hasn't scrolled up
        return chat.yview()[1] > 0.97

    def put(text, tag):
        stick = follow()
        pre = "\n" if ai["open"] else ""                  # leading newline: AI text may still be typing above
        chat.config(state="normal")
        chat.insert("end", pre + text + "\n", tag)
        chat.config(state="disabled")
        if stick:
            chat.see("end")
        blocks.append({"tag": tag, "text": pre + text + "\n"})
        notify_readers()

    def ins_at_ai_end(text, tag):
        stick = follow()
        chat.config(state="normal")
        chat.insert("aiend", text, tag)
        chat.mark_set("aiend", f"aiend+{len(text)}c")
        chat.config(state="disabled")
        if stick:
            chat.see("end")
        blk = ai.get("block")
        if blk is not None:
            if tag == "ai":
                blk["text"] += text
            else:                                         # e.g. "(stopped)": goes right after the answer block
                pos = next((i for i, b in enumerate(blocks) if b is blk), len(blocks) - 1)
                blocks.insert(pos + 1, {"tag": tag, "text": text})
        notify_readers()

    def finish_ai():
        ins_at_ai_end("\n", "ai")
        stopped = bool(ai["end"] and ai["end"].get("stopped"))
        if stopped:
            ins_at_ai_end("(stopped)\n", "sys")
        ai.update(open=False, end=None)
        if settings.get("tts_enabled") and not stopped:
            text = ai["target"]
            root.after(1000, lambda t=text: tts.speak(t))

    def flush_ai():                                       # show everything right now and close the block
        rest = ai["target"][len(ai["shown"]):] if ai["target"].startswith(ai["shown"]) else None
        if rest:
            ins_at_ai_end(rest, "ai")
            ai["shown"] = ai["target"]
        finish_ai()

    def reveal_tick():
        now = time.time()
        dt, ai["t"] = now - ai["t"], now
        if ai["open"]:
            tgt = ai["target"]
            if not tgt.startswith(ai["shown"]):           # text changed under us (hidden reasoning removed): redraw
                chat.config(state="normal")
                chat.delete("aistart", "aiend")
                chat.mark_set("aiend", "aistart")
                chat.config(state="disabled")
                ai["shown"] = ""
                if ai.get("block") is not None:
                    ai["block"]["text"] = ""
            missing = len(tgt) - len(ai["shown"])
            if missing > 0:
                cps = SPEED_CPS[int(settings["text_speed"])]
                if cps is None or (ai["end"] and ai["end"].get("stopped")):
                    n = missing
                else:
                    ai["budget"] += cps * dt
                    n = min(missing, int(ai["budget"]))
                    ai["budget"] -= n
                if n > 0:
                    chunk = tgt[len(ai["shown"]):len(ai["shown"]) + n]
                    ins_at_ai_end(chunk, "ai")
                    ai["shown"] += chunk
            else:
                ai["budget"] = 0.0
                if ai["end"] is not None:
                    finish_ai()
        root.after(30, reveal_tick)

    class Reader:
        """A separate window that mirrors the conversation, with its own font size. Drag it to another monitor."""

        def __init__(self):
            self.win = tk.Toplevel(root)
            self.win.title("Voice Chat - Reader")
            self.win.geometry(f"640x560+{root.winfo_rootx() + 60}+{root.winfo_rooty() + 60}")
            self.win.minsize(320, 200)
            self.size = max(18, fs["size"])
            self.latest = tk.BooleanVar(value=False)
            self.ontop = tk.BooleanVar(value=False)
            tb = ttk.Frame(self.win)
            tb.pack(fill="x", padx=8, pady=(6, 0))
            ttk.Button(tb, text="A-", width=3, command=lambda: self.bump(-2)).pack(side="left")
            ttk.Button(tb, text="A+", width=3, command=lambda: self.bump(2)).pack(side="left", padx=(2, 10))
            ttk.Checkbutton(tb, text="Latest answer only", variable=self.latest, command=self.render).pack(side="left")
            ttk.Checkbutton(tb, text="Always on top", variable=self.ontop,
                            command=lambda: self.win.attributes("-topmost", bool(self.ontop.get()))).pack(side="left", padx=(10, 0))
            ttk.Button(tb, text="Close", command=self.close).pack(side="right")
            body = ttk.Frame(self.win)
            body.pack(fill="both", expand=True)
            self.text = tk.Text(body, wrap="word", bd=0, padx=16, pady=10, state="disabled", cursor="arrow", spacing3=8, height=4)
            sb = ttk.Scrollbar(body, command=self.text.yview)
            self.text.config(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
            self.text.pack(side="left", fill="both", expand=True)
            self.win.bind("<Control-MouseWheel>", lambda e: self.bump(2 if e.delta > 0 else -2))
            self.win.protocol("WM_DELETE_WINDOW", self.close)
            self.restyle()

        def restyle(self):
            try:
                self.win.configure(bg=T["bg"])
                self.text.config(bg=T["bg"], fg=T["fg"])
                setup_tags(self.text, settings["font"], self.size)
            except tk.TclError:
                return
            self.render()

        def bump(self, d):
            self.size = max(12, min(80, self.size + d))
            self.restyle()

        def visible(self):
            if not self.latest.get():
                return blocks
            last = max((i for i, b in enumerate(blocks) if b["tag"] == "you"), default=0)
            return blocks[last:]

        def render(self):
            try:
                first, last = self.text.yview()
                self.text.config(state="normal")
                self.text.delete("1.0", "end")
                for b in self.visible():
                    self.text.insert("end", b["text"], b["tag"])
                self.text.config(state="disabled")
                if last > 0.97:
                    self.text.see("end")                  # was at the bottom: keep following
                else:
                    self.text.yview_moveto(first)         # reader scrolled up: leave them where they were
            except tk.TclError:
                pass

        def close(self):
            if self in readers:
                readers.remove(self)
            try:
                self.win.destroy()
            except tk.TclError:
                pass

    def pop_out():
        readers.append(Reader())

    # ----- bottom bars: typed input + buttons (always visible) and the ☰ Menu (hidden until opened)
    bar = ttk.Frame(root)
    bar.pack(fill="x", padx=8, pady=6)
    menu = ttk.Frame(root)                                # quick controls + settings; shown by the ☰ Menu button
    bar2 = ttk.Frame(menu)
    bar2.pack(side="bottom", fill="x", padx=8, pady=(4, 0))

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
    ttk.Button(bar, text="New session", command=lambda: threading.Thread(target=engine.new_session, daemon=True).start()).pack(side="left", padx=(6, 0))
    ttk.Button(bar, text="End & save", command=lambda: threading.Thread(target=engine.end_now, daemon=True).start()).pack(side="left", padx=(6, 0))
    ttk.Button(bar, text="Stop answer", command=lambda: (engine.abort.set(), tts.stop())).pack(side="left", padx=(6, 0))
    ttk.Button(bar, text="A-", width=3, command=lambda: set_font_size(-2)).pack(side="left", padx=(6, 0))
    ttk.Button(bar, text="A+", width=3, command=lambda: set_font_size(2)).pack(side="left", padx=(2, 0))
    ttk.Button(bar, text="Pop out", command=pop_out).pack(side="left", padx=(6, 0))
    btn_set = ttk.Button(bar, text="☰ Menu")
    btn_set.pack(side="left", padx=(6, 0))

    def set_font_size(d):
        fs["size"] = max(12, min(60, fs["size"] + d))
        settings["font_size"] = fs["size"]
        save_settings(settings)
        apply_font()

    def clear_chat():
        chat.config(state="normal")
        chat.delete("1.0", "end")
        chat.config(state="disabled")
        ai.update(open=False, target="", shown="", end=None, block=None)
        blocks.clear()
        notify_readers()

    def toggle_pause():
        if engine.ended:
            return
        engine.paused = not engine.paused
        put("Paused." if engine.paused else "Listening again.", "sys")
        engine._update_state()

    btn_pause.config(command=toggle_pause)

    # text-speed slider (+ one-click theme / font cycling) on its own thin bar
    ttk.Label(bar2, text="Text speed").pack(side="left")
    v_speed = tk.DoubleVar(value=settings["text_speed"])
    lbl_speed = ttk.Label(bar2, text=speed_label(settings["text_speed"]), width=16)

    def on_speed(val):
        n = max(1, min(10, int(round(float(val)))))
        lbl_speed.config(text=speed_label(n))
        if n != settings["text_speed"]:
            settings["text_speed"] = n
            save_settings(settings)

    ttk.Label(bar2, text="slow").pack(side="left", padx=(8, 2))
    ttk.Scale(bar2, from_=1, to=10, variable=v_speed, command=on_speed, length=260).pack(side="left")
    ttk.Label(bar2, text="fast").pack(side="left", padx=(2, 8))
    lbl_speed.pack(side="left")
    ttk.Label(bar2, text="Mode").pack(side="left", padx=(18, 4))
    v_backend = tk.StringVar(value=settings["backend"])
    cb_backend = ttk.Combobox(bar2, textvariable=v_backend, values=BACKENDS, state="readonly", width=22)
    cb_backend.pack(side="left")
    combos.append(cb_backend)

    def on_backend(_e=None):
        engine.set_backend(v_backend.get())                # refuses (and says why) if Hermes is missing
        root.after(150, lambda: (v_backend.set(settings["backend"]), sync_quick()))

    cb_backend.bind("<<ComboboxSelected>>", on_backend)
    btn_font = ttk.Button(bar2, text="Font ▸")
    btn_font.pack(side="right")
    btn_theme = ttk.Button(bar2, text="Theme ▸")
    btn_theme.pack(side="right", padx=(0, 6))

    # ----- settings panel
    sp = ttk.Frame(menu)
    sp.pack(side="top", fill="x")
    v = {k: tk.StringVar(value=str(settings[k])) for k in ("ollama_model", "whisper_model", "language", "mic", "wake_word")}
    v_wake = tk.BooleanVar(value=settings["wake_enabled"])
    v_sil = tk.DoubleVar(value=settings["silence_s"])
    v_sens = tk.DoubleVar(value=settings["sensitivity"])
    v_sess = tk.StringVar(value=str(settings["session_min"]))
    v_ram = tk.StringVar(value=str(settings["ram_limit"]))
    v_theme = tk.StringVar(value=settings["theme"])
    v_font = tk.StringVar(value=settings["font"])

    def current_preset_name():
        p = settings["system_prompt"].strip()
        return next((n for n, t in PROMPT_PRESETS.items() if t.strip() == p), CUSTOM_PRESET)

    v_preset = tk.StringVar(value=current_preset_name())

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

    cb_model = ttk.Combobox(sp, textvariable=v["ollama_model"], values=models0, width=34)
    cb_mic = ttk.Combobox(sp, textvariable=v["mic"], values=mic_names(), width=34, state="readonly")
    cb_wh = ttk.Combobox(sp, textvariable=v["whisper_model"], values=WHISPER_MODELS, width=16, state="readonly")
    cb_lang = ttk.Combobox(sp, textvariable=v["language"], values=LANGUAGES, width=16)
    cb_theme = ttk.Combobox(sp, textvariable=v_theme, values=list(THEMES), width=34, state="readonly")
    cb_font = ttk.Combobox(sp, textvariable=v_font, values=fonts_ok, width=16, state="readonly")
    cb_preset = ttk.Combobox(sp, textvariable=v_preset, values=list(PROMPT_PRESETS) + [CUSTOM_PRESET], state="readonly", width=44)
    combos.extend([cb_model, cb_mic, cb_wh, cb_lang, cb_theme, cb_font, cb_preset])
    row(0, 0, "Chat model (chat mode)", cb_model)
    row(1, 0, "Microphone", cb_mic)
    row(0, 1, "Speech model", cb_wh)
    row(1, 1, "Language", cb_lang)
    row(2, 0, "Silence before sending (s)", ttk.Scale(sp, from_=0.8, to=4.0, variable=v_sil))
    row(2, 1, "Sensitivity", ttk.Scale(sp, from_=0.2, to=0.9, variable=v_sens))
    row(3, 0, "Session length (minutes)", ttk.Spinbox(sp, from_=1, to=60, textvariable=v_sess, width=8))
    row(3, 1, "RAM guard trips at (%)", ttk.Spinbox(sp, from_=70, to=97, textvariable=v_ram, width=8))
    wf = ttk.Frame(sp)
    ttk.Checkbutton(wf, text="Require wake word:", variable=v_wake).pack(side="left")
    ttk.Entry(wf, textvariable=v["wake_word"], width=14).pack(side="left", padx=6)
    wf.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=3)
    ttk.Button(sp, text="Refresh models", command=lambda: fetch_models_async()).grid(row=4, column=2, sticky="w", padx=8)
    row(5, 0, "Color theme", cb_theme)
    row(5, 1, "Font", cb_font)
    ttk.Label(sp, text="Mentor style").grid(row=6, column=0, sticky="w", padx=8, pady=3)
    cb_preset.grid(row=6, column=1, columnspan=2, sticky="w", padx=(0, 8), pady=3)
    ttk.Label(sp, text="System prompt").grid(row=7, column=0, sticky="nw", padx=8, pady=3)
    txt_prompt = reg(tk.Text(sp, height=4, wrap="word", bd=0, font=("Segoe UI", 10)), bg="panel", fg="fg", insertbackground="fg")
    txt_prompt.insert("1.0", settings["system_prompt"])
    txt_prompt.grid(row=7, column=1, columnspan=3, sticky="ew", padx=8, pady=3)
    v_amodel = tk.StringVar(value=settings["hermes_model"])
    v_tools = tk.StringVar(value=settings["hermes_toolsets"])
    v_reason = tk.StringVar(value=settings["hermes_reasoning"])
    cb_amodel = ttk.Combobox(sp, textvariable=v_amodel, values=models0, width=34)
    cb_reason = ttk.Combobox(sp, textvariable=v_reason, values=["none", "low", "medium", "high"], state="readonly", width=16)
    combos.extend([cb_amodel, cb_reason])
    row(8, 0, "Agent model (agent mode)", cb_amodel)
    row(8, 1, "Agent tools", ttk.Entry(sp, textvariable=v_tools, width=20))
    row(9, 0, "Agent reasoning", cb_reason)
    ttk.Label(sp, text="Agent tools: 'web' = search only (safe). terminal / file / code_execution can change this PC.").grid(
        row=10, column=0, columnspan=4, sticky="w", padx=8)
    ttk.Label(sp, text="Agent instruction").grid(row=11, column=0, sticky="nw", padx=8, pady=3)
    txt_aprefix = reg(tk.Text(sp, height=3, wrap="word", bd=0, font=("Segoe UI", 10)), bg="panel", fg="fg", insertbackground="fg")
    txt_aprefix.insert("1.0", settings["hermes_prefix"])
    txt_aprefix.grid(row=11, column=1, columnspan=3, sticky="ew", padx=8, pady=3)
    sp.columnconfigure(1, weight=1)
    sp.columnconfigure(3, weight=1)

    # TTS row (row 12)
    v_tts       = tk.BooleanVar(value=bool(settings.get("tts_enabled", False)))
    _kok_init   = TTS_KOKORO_LABELS.get(settings.get("tts_voice", "af_heart"), "Heart (US ♀)")
    v_tts_voice = tk.StringVar(value=_kok_init)
    tts_frame   = ttk.Frame(sp)
    ttk.Checkbutton(tts_frame, text="Read answers aloud", variable=v_tts).pack(side="left")
    ttk.Label(tts_frame, text="  Voice:").pack(side="left")
    cb_tts_voice = ttk.Combobox(tts_frame, textvariable=v_tts_voice,
                                values=list(TTS_KOKORO_LABELS.values()), state="readonly", width=14)
    cb_tts_voice.pack(side="left", padx=(4, 8))
    combos.append(cb_tts_voice)
    ttk.Button(tts_frame, text="Stop speaking", command=lambda: tts.stop()).pack(side="left", padx=(4, 0))
    tts_frame.grid(row=12, column=0, columnspan=4, sticky="w", padx=8, pady=3)

    def settings_changed(*_):
        settings["ollama_model"] = v["ollama_model"].get().strip()
        settings["language"] = v["language"].get().strip() or "en"
        settings["wake_word"] = v["wake_word"].get().strip() or "computer"
        settings["wake_enabled"] = v_wake.get()
        settings["silence_s"] = round(v_sil.get(), 2)
        settings["sensitivity"] = round(v_sens.get(), 2)
        settings["system_prompt"] = txt_prompt.get("1.0", "end").strip() or DEFAULT_PROMPT
        settings["prompt_preset"] = current_preset_name()
        v_preset.set(settings["prompt_preset"])
        settings["hermes_model"] = v_amodel.get().strip() or settings["hermes_model"]
        settings["hermes_toolsets"] = ",".join(t for t in re.split(r"[,\s]+", v_tools.get()) if t) or "web"
        settings["hermes_reasoning"] = v_reason.get() or "none"
        settings["hermes_prefix"] = txt_aprefix.get("1.0", "end").strip() or DEFAULTS["hermes_prefix"]
        for key, var, lo, hi in (("session_min", v_sess, 1, 60), ("ram_limit", v_ram, 70, 97)):
            try:
                settings[key] = max(lo, min(hi, int(float(var.get()))))     # ignore half-typed values
            except ValueError:
                pass
        settings["tts_enabled"] = v_tts.get()
        raw_voice = _TTS_KOKORO_LABEL_TO_KEY.get(v_tts_voice.get())
        if raw_voice:
            settings["tts_voice"] = raw_voice
            tts.set_voice(raw_voice)
        save_settings(settings)

    for var in (v["ollama_model"], v["language"], v["wake_word"], v_wake, v_sil, v_sens, v_sess, v_ram):
        var.trace_add("write", settings_changed)
    txt_prompt.bind("<FocusOut>", settings_changed)
    for var in (v_amodel, v_tools, v_reason):
        var.trace_add("write", settings_changed)
    txt_aprefix.bind("<FocusOut>", settings_changed)
    v_tts.trace_add("write", settings_changed)
    cb_tts_voice.bind("<<ComboboxSelected>>", settings_changed)

    def on_preset(_=None):
        name = v_preset.get()
        if name in PROMPT_PRESETS:                        # "Custom (edited)" just keeps whatever is in the box
            txt_prompt.delete("1.0", "end")
            txt_prompt.insert("1.0", PROMPT_PRESETS[name])
            settings_changed()
            put(f"Mentor style: {name}. Applies from your next question - say 'new session' for a clean switch.", "sys")

    cb_preset.bind("<<ComboboxSelected>>", on_preset)

    # ----- theme + font switching (live, no restart)
    def apply_theme(name):
        if name not in THEMES:
            return
        settings["theme"] = name
        v_theme.set(name)
        save_settings(settings)
        T.clear()
        T.update(THEMES[name])
        style_ttk()
        for widget, opts in registry:
            try:
                widget.config(**{k: T[role] for k, role in opts.items()})
            except tk.TclError:
                pass
        apply_font()
        set_banner()
        set_timer(*cur["timer"])
        for sk in sparks.values():
            if sk.last:
                sk.draw(*sk.last)

    def apply_font_name(name):
        if name in fonts_ok:
            settings["font"] = name
            v_font.set(name)
            save_settings(settings)
            apply_font()

    cb_theme.bind("<<ComboboxSelected>>", lambda _e: apply_theme(v_theme.get()))
    cb_font.bind("<<ComboboxSelected>>", lambda _e: apply_font_name(v_font.get()))
    btn_theme.config(command=lambda: apply_theme(list(THEMES)[(list(THEMES).index(settings["theme"]) + 1) % len(THEMES)]))
    btn_font.config(command=lambda: apply_font_name(fonts_ok[(fonts_ok.index(settings["font"]) + 1) % len(fonts_ok)]))

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

    def show_menu(on):
        settings["menu_open"] = bool(on)
        save_settings(settings)
        if on:
            menu.pack(fill="x", before=bar)
        else:
            menu.pack_forget()

    btn_set.config(command=lambda: show_menu(not settings["menu_open"]))
    if settings["menu_open"]:
        show_menu(True)

    # ----- model lists (they refresh by themselves) + the quick Model box that follows the current mode
    models_state = {"ok": bool(models0)}

    def apply_models(lst):
        for cb in (cb_quick, cb_model, cb_amodel):
            cb.config(values=lst)

    def fetch_models_async():
        threading.Thread(target=lambda: evq.put(("models", {"list": list_models(timeout=2)})), daemon=True).start()

    def poll_models(n=0):                                 # Ollama may still be starting when the window opens: keep trying
        if not models_state["ok"] and n < 80:
            fetch_models_async()
            root.after(2500, lambda: poll_models(n + 1))

    def sync_quick(*_):
        agent = engine.agent_mode()
        lbl_quick.config(text="Agent model" if agent else "Chat model")
        # read the live variables, not `settings`: Tk fires a variable's callbacks newest-first, so this can run before
        # settings_changed has stored the new value
        v_quick.set(v_amodel.get().strip() if agent else v["ollama_model"].get().strip())

    def on_quick(_e=None):
        name = v_quick.get().strip()
        if not name:
            return
        if engine.agent_mode():
            if name != settings["hermes_model"]:
                v_amodel.set(name)
                put(f"Agent model -> {name} (a 64K copy is created automatically on first use if needed)", "sys")
        elif name != settings["ollama_model"]:
            v["ollama_model"].set(name)
            put(f"Chat model -> {name}", "sys")

    cb_quick.bind("<<ComboboxSelected>>", on_quick)
    cb_quick.bind("<Return>", on_quick)
    for cb in (cb_quick, cb_model, cb_amodel):
        cb.config(postcommand=fetch_models_async)         # opening any model list also refreshes it
    v["ollama_model"].trace_add("write", sync_quick)
    v_amodel.trace_add("write", sync_quick)

    # ----- engine events -> widgets
    def handle(kind, kw):
        if kind == "state":
            cur["state"] = kw["name"]
            set_banner()
            btn_pause.config(text="Resume" if kw["name"] == "paused" else "Pause")
        elif kind == "level":
            level["value"] = kw["p"] * 100
        elif kind == "timer":
            r = kw["remaining"]
            if r is None:
                set_timer(f"Session {settings['session_min']} min - starts when you speak", "muted")
            else:
                r = max(0, int(r))
                set_timer(f"Session  {r // 60:02d}:{r % 60:02d} left", "warn" if r <= 60 else "fg")
        elif kind == "user":
            put("You:  " + kw["text"], "you")
        elif kind == "ai_start":
            tts.stop()
            if ai["open"]:
                flush_ai()                                # previous answer was still being typed out
            chat.config(state="normal")
            chat.mark_set("aistart", "end-1c")
            chat.mark_gravity("aistart", "left")
            chat.mark_set("aiend", "end-1c")
            chat.mark_gravity("aiend", "left")
            chat.config(state="disabled")
            blk = {"tag": "ai", "text": ""}
            blocks.append(blk)
            ai.update(open=True, target="", shown="", end=None, budget=0.0, block=blk)
        elif kind == "ai_set":
            ai["target"] = kw["text"] if kw["text"] or not kw["thinking"] else "(thinking...)"
        elif kind == "ai_end":
            if ai["open"]:
                ai["end"] = {"stopped": bool(kw.get("stopped"))}
        elif kind == "session_end":
            if ai["open"]:
                flush_ai()
            set_timer("Session over", "muted")
            put(f"SESSION ENDED ({kw['reason']}).", "ok")
            put(f"Saved: {kw['path']}" if kw["path"] else "Nothing to save.", "ok")
            put("Say \"new session\" (or click New session) to start another.", "sys")
        elif kind == "system":
            put(kw["text"], "sys")
        elif kind in ("error", "alert"):
            put(("!! " if kind == "alert" else "") + kw["text"], "err")
        elif kind == "ignored":
            put("(ignored: " + kw["text"] + ")", "ign")
        elif kind == "backend":
            v_backend.set(kw["name"])
            save_settings(settings)
            sync_quick()
        elif kind == "models":
            if kw["list"]:
                models_state["ok"] = True
                apply_models(kw["list"])
                if not settings["ollama_model"]:
                    settings["ollama_model"] = pick_default_model(kw["list"])
                    v["ollama_model"].set(settings["ollama_model"])
                    save_settings(settings)
        elif kind == "agent_model":
            v_amodel.set(kw["name"])
        elif kind == "clear":
            clear_chat()

    def poll():
        try:
            while True:
                handle(*evq.get_nowait())
        except queue.Empty:
            pass
        except Exception:  # noqa: BLE001 — never let a handle() crash kill the loop
            import traceback
            traceback.print_exc()
        root.after(40, poll)

    # ----- go
    style_ttk()                                           # widgets created after the first call (comboboxes) pick up the palette
    set_banner()
    models = models0
    if not settings["ollama_model"] and models:
        settings["ollama_model"] = pick_default_model(models)
        v["ollama_model"].set(settings["ollama_model"])
        save_settings(settings)
    engine.emit("state", name="loading")
    engine.emit("timer", remaining=None)
    engine.start()
    start_mic()
    sync_quick()
    poll_models()
    poll()
    reveal_tick()
    root.after(1500, draw_monitor)

    def on_close():
        engine.stop()
        tts.stop()
        engine.save_session()                             # never lose notes when closing mid-session
        monitor.stop()
        if stream["s"]:
            stream["s"].close()
        root.destroy()
        engine.release_models()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
