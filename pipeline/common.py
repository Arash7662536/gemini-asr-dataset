"""Shared utilities: config, jsonl IO, audio decoding, Persian text normalization."""
import json
import os
import re
import subprocess

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# config / IO
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


DRY_RUN_CALLS = 10  # stage-1 call cap when --dry-run is set without --limit


def add_common_args(ap):
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="smoke test: small sample, separate <work_dir>_dryrun")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap items processed by this stage (0 = no cap)")


def load_config_args(args):
    """load_config + CLI overrides; --dry-run redirects work_dir so real outputs stay clean."""
    cfg = load_config(args.config)
    if getattr(args, "dry_run", False):
        cfg["paths"]["work_dir"] = cfg["paths"]["work_dir"].rstrip("/") + "_dryrun"
    return cfg


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class JsonlWriter:
    """Append-mode jsonl writer, flushed per row so stages are resumable."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def write(self, obj):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def done_ids(path, key):
    """IDs already present in an output jsonl (for resume)."""
    if not os.path.exists(path):
        return set()
    return {r[key] for r in read_jsonl(path)}


def done_ok_ids(path, key="chunk_id"):
    """IDs whose LAST row has ok=true — errored rows are retried on re-run."""
    if not os.path.exists(path):
        return set()
    last = {}
    for r in read_jsonl(path):
        last[r[key]] = bool(r.get("ok"))
    return {k for k, ok in last.items() if ok}


def read_dict_jsonl(path, key="chunk_id"):
    """jsonl as {key: row}; last occurrence wins (crash-safe dedup)."""
    out = {}
    if os.path.exists(path):
        for r in read_jsonl(path):
            out[r[key]] = r
    return out


def read_chunks(work_dir):
    return read_dict_jsonl(os.path.join(work_dir, "chunks.jsonl"), "chunk_id")


def parse_shard(s):
    i, n = s.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise ValueError(f"bad shard spec: {s}")
    return i, n


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def decode_mono(path, sr=16000):
    """Decode any audio file to float32 mono at sr (ffmpeg, container-agnostic)."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sr), "-",
    ]
    out = subprocess.run(cmd, capture_output=True).stdout
    if not out:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(out, dtype="<i2").astype(np.float32) / 32768.0


def rms(x):
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def envelope(x, frame):
    """Mean-abs envelope, one value per `frame` samples."""
    n = len(x) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.abs(x[: n * frame]).reshape(n, frame).mean(axis=1)


# ---------------------------------------------------------------------------
# call filename parsing (Asterisk-style leg recordings)
# ---------------------------------------------------------------------------

# external-603-1001-20260518-134216-1779100017.89520636_r.wav
FNAME_RE = re.compile(
    r"^(?P<prefix>[A-Za-z]+)-(?P<ext>\d+)-(?P<line>\d+)-"
    r"(?P<date>\d{8})-(?P<time>\d{6})-(?P<uid>[\d.]+)_(?P<leg>[rt])\.wav$"
)


def parse_leg_filename(name):
    """Returns (call_id, leg, meta) — call_id is the shared stem of both legs."""
    m = FNAME_RE.match(name)
    if m:
        call_id = name[: -len(f"_{m['leg']}.wav")]
        meta = {"extension": m["ext"], "line": m["line"],
                "date": m["date"], "time": m["time"], "uid": m["uid"]}
        return call_id, m["leg"], meta
    base = name[:-4] if name.endswith(".wav") else name
    leg = None
    if base.endswith(("_r", "_t")):
        leg = base[-1]
        base = base[:-2]
    return base, leg, {"extension": "unknown", "line": "", "date": "", "time": "", "uid": base}


# ---------------------------------------------------------------------------
# Persian text normalization
# ---------------------------------------------------------------------------

ZWNJ = "‌"

_CHAR_MAP = str.maketrans({
    # Arabic -> Persian letters
    "ي": "ی",  # ي -> ی
    "ى": "ی",  # ى -> ی
    "ے": "ی",  # ے -> ی
    "ك": "ک",  # ك -> ک
    "أ": "ا",  # أ -> ا
    "إ": "ا",  # إ -> ا
    "ٱ": "ا",  # ٱ -> ا
    "ة": "ه",  # ة -> ه
    "ھ": "ه",  # ھ -> ه
    # Arabic-Indic -> Persian digits
    "٠": "۰", "١": "۱", "٢": "۲",
    "٣": "۳", "٤": "۴", "٥": "۵",
    "٦": "۶", "٧": "۷", "٨": "۸",
    "٩": "۹",
})

# tashkeel, dagger alif, tatweel
_DIACRITICS = re.compile("[ً-ْٰـ]")
_WS = re.compile(r"\s+")


def normalize_train(text):
    """Light normalization applied to training labels (keeps punctuation, ZWNJ)."""
    t = text.translate(_CHAR_MAP)
    t = _DIACRITICS.sub("", t)
    t = re.sub(f"{ZWNJ}+", ZWNJ, t)
    t = re.sub(f"(?: {ZWNJ})|(?:{ZWNJ} )", " ", t)
    return _WS.sub(" ", t).strip()


_PUNCT = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~،؛؟«»…]")
_FA2EN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹",
                              "0123456789")


def normalize_compare(text):
    """Aggressive normalization used only for CER comparison (retry agreement)."""
    t = normalize_train(text).replace(ZWNJ, " ")
    t = _PUNCT.sub(" ", t)
    t = t.translate(_FA2EN_DIGITS).lower()
    return _WS.sub(" ", t).strip()


def latin_ratio(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    latin = sum(1 for c in chars if ("a" <= c <= "z") or ("A" <= c <= "Z"))
    return latin / len(chars)


def has_repetition(text):
    """Consecutive repeated n-grams — the classic ASR/LLM hallucination shape."""
    words = text.split()
    for n, times in ((1, 6), (2, 4), (3, 3)):
        if len(words) < n * times:
            continue
        for i in range(len(words) - n * times + 1):
            gram = words[i:i + n]
            if all(words[i + k * n: i + (k + 1) * n] == gram for k in range(1, times)):
                return True
    return False


def edit_distance(a, b):
    """Levenshtein over two sequences (lists or strings)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    """Character error rate on aggressively normalized text."""
    r = normalize_compare(ref).replace(" ", "")
    h = normalize_compare(hyp).replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)
