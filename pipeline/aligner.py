"""CTC forced alignment core, shared by stage 4 (align) and stage 5 (rescue).

Backend: the GitHub package MahmoudAshraf97/ctc-forced-aligner (torch), which
loads the HuggingFace checkpoint MahmoudAshraf/mms-300m-1130-forced-aligner —
Meta's MMS-300M fine-tuned for alignment over a romanized (uroman) vocabulary,
Persian = ISO 639-3 "fas". Install:

    pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

Word confidences are geometric means of per-frame alignment scores, a real
[0, 1] confidence. The Viterbi kernel here replaces the package's C++
align_sequences, whose backpointer buffer is undersized for tight alignments
(audio barely longer than the text) — it corrupts the heap and aborts the
process. Same inputs/outputs and tie-breaking; raises ValueError when no
alignment exists.
"""
import math

import numpy as np


def load_aligner(cfg):
    """Returns (emit_fn, tokenizer, cfa_module, label)."""
    import ctc_forced_aligner as cfa

    if not hasattr(cfa, "load_alignment_model"):
        raise SystemExit(
            "the installed `ctc_forced_aligner` is the PyPI/ONNX package, not the\n"
            "torch one this pipeline uses. Install the GitHub package:\n"
            "  pip uninstall -y ctc-forced-aligner\n"
            "  pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git"
        )
    import torch
    device = cfg["runtime"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model, tokenizer = cfa.load_alignment_model(device, dtype=dtype)
    batch_size = cfg["stage4"]["batch_size"]

    def emit(wav):
        audio = torch.from_numpy(np.ascontiguousarray(wav)).to(device=device, dtype=dtype)
        return cfa.generate_emissions(model, audio, batch_size=batch_size)

    return emit, tokenizer, cfa, f"torch:{device}"


def _viterbi_forced_align(log_probs, targets, blank):
    """CTC forced alignment (Viterbi over the 2L+1 blank-interleaved states).

    log_probs: (T, C) float; targets: (L,) int; returns (path (T,), scores (T,)).
    """
    log_probs = np.asarray(log_probs, dtype=np.float32)
    T = log_probs.shape[0]
    L = len(targets)
    if L == 0:
        raise ValueError("alignment_infeasible: empty targets")
    S = 2 * L + 1
    labels = np.full(S, blank, dtype=np.int64)
    labels[1::2] = targets
    allow_skip = np.zeros(S, dtype=bool)        # i-2 transition: odd states whose
    allow_skip[3::2] = targets[1:] != targets[:-1]  # token differs from the previous
    NEG = np.float32(-np.inf)

    alpha = np.full(S, NEG, dtype=np.float32)
    alpha[0] = log_probs[0, blank]
    if S > 1:
        alpha[1] = log_probs[0, targets[0]]
    bp = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        x0 = alpha
        x1 = np.concatenate(([NEG], alpha[:-1]))
        x2 = np.concatenate(([NEG, NEG], alpha[:-2]))
        x2 = np.where(allow_skip, x2, NEG)
        # same tie-breaking as the C++: skip only if strictly best, then step
        choose2 = (x2 > x1) & (x2 > x0)
        choose1 = ~choose2 & (x1 > x0) & (x1 > x2)
        best = np.where(choose2, x2, np.where(choose1, x1, x0))
        bp[t] = np.where(choose2, 2, np.where(choose1, 1, 0))
        alpha = best + log_probs[t, labels]

    s = S - 1 if (S > 1 and alpha[S - 1] > alpha[S - 2]) else max(S - 2, 0)
    if not np.isfinite(alpha[s]):
        raise ValueError(
            f"alignment_infeasible: no valid CTC path (targets {L}, frames {T})")
    states = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        states[t] = s
        # int(): keep s a Python int. numpy 2.x (NEP 50) would otherwise promote
        # `python_int - np.int8` to int8 and overflow once S > 127 (long texts).
        s -= int(bp[t, s])
    path = labels[states]
    scores = log_probs[np.arange(T), path].astype(np.float32)
    return path, scores


def get_alignments_safe(cfa, emissions, tokens, tokenizer):
    """Drop-in for the package's get_alignments, minus the crashing C++ kernel."""
    if hasattr(emissions, "cpu"):
        emissions = emissions.float().cpu().numpy()
    emissions = np.asarray(emissions, dtype=np.float32)
    if emissions.ndim == 3:
        emissions = emissions.reshape(-1, emissions.shape[-1])

    dictionary = {str(k).lower(): v for k, v in tokenizer.get_vocab().items()}
    dictionary["<star>"] = len(dictionary)
    token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    if not token_indices:
        raise ValueError("alignment_infeasible: no tokens in vocabulary")
    blank_id = dictionary.get("<blank>", getattr(tokenizer, "pad_token_id", 0))

    path, scores = _viterbi_forced_align(
        emissions, np.asarray(token_indices, dtype=np.int64), blank_id)
    idx_to_token = {v: k for k, v in dictionary.items()}
    segments = cfa.merge_repeats(path.tolist(), idx_to_token)
    return segments, scores, idx_to_token[blank_id]


def align_words(cfa, emit, tokenizer, wav, text, lang):
    """Run one chunk; returns list of {t, s, e, c, scored} word dicts."""
    emissions, stride = emit(wav)
    tokens_starred, text_starred = cfa.preprocess_text(text, romanize=True, language=lang)
    # words that romanize to nothing (digits like ۱۲۵۳۴) would get degenerate
    # spans; align them as <star> instead so the spoken number's audio is
    # absorbed with real timing — but exclude them from confidence scoring
    unscored = set()
    for i, (tok, txt) in enumerate(zip(tokens_starred, text_starred)):
        if txt != "<star>" and not str(tok).strip():
            tokens_starred[i] = "<star>"
            unscored.add(i)
    vocab = {str(k).lower() for k in tokenizer.get_vocab()}
    vocab.add("<star>")
    flat = [c for c in " ".join(tokens_starred).split(" ") if c in vocab]
    need = len(flat) + sum(1 for a, b in zip(flat, flat[1:]) if a == b)
    n_frames = int(emissions.shape[0])
    if n_frames < need:
        raise ValueError(
            f"alignment_infeasible: targets need {need} frames, audio has {n_frames}")
    segments, scores, blank_token = get_alignments_safe(cfa, emissions, tokens_starred, tokenizer)
    spans = cfa.get_spans(tokens_starred, segments, blank_token)
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    log_domain = bool(scores.min() < -1e-3)

    words = []
    for i, t in enumerate(text_starred):
        if t == "<star>" or i >= len(spans) or not spans[i]:
            continue
        f0, f1 = int(spans[i][0].start), int(spans[i][-1].end)
        if f1 <= f0 or f1 > len(scores):
            continue
        m = float(scores[f0:f1].mean())
        conf = math.exp(min(m, 0.0)) if log_domain else min(max(m, 0.0), 1.0)
        words.append({"t": t, "s": f0 * stride / 1000.0, "e": f1 * stride / 1000.0,
                      "c": conf, "scored": i not in unscored})
    return words


def align_text(handle, wav, text, lang):
    """Align one transcript against one waveform; never raises.

    Returns a metrics dict with ok/error and the fields stage 6 filters on.
    """
    emit, tokenizer, cfa, _ = handle
    row = {"ok": False, "ctc_align_score": 0.0, "ctc_align_min_word": 0.0,
           "n_words": 0, "first_word_start": None, "last_word_end": None,
           "error": None}
    try:
        words = align_words(cfa, emit, tokenizer, wav, text, lang)
        confs = [w["c"] for w in words if w["scored"]]
        if not words:
            row["error"] = "no_aligned_words"
        elif not confs:
            row["error"] = "no_scorable_words"
        else:
            row.update({
                "ok": True,
                "ctc_align_score": round(float(np.mean(confs)), 4),
                "ctc_align_min_word": round(float(np.min(confs)), 4),
                "n_words": len(words),
                "n_unscored_words": len(words) - len(confs),
                "first_word_start": round(words[0]["s"], 3),
                "last_word_end": round(words[-1]["e"], 3),
                "word_scores": [round(x, 3) for x in confs],
            })
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row
