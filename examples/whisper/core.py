# Derived from Kevin Bondzio, MIT, weight-patching@d7df0d9bd15afb952b1e0cb64a067806563aa8e3; see UPSTREAM-LICENSE.
"""Whisper helpers for tasks/whisper_silence.py (encoder-decoder, so not patchkit/llm.py).

Copied and trimmed from the original experiment (whisper_halluc/common.py) so this repo stands alone:
- data: 16 kHz mono float32 clips in $WHISPER_SILENCE_DATA/derived/<suite>.npy + <suite>.json manifests
  (default ../data_audio relative to the weight-patching root; rebuild with tasks/_whisper_prepare_data.py)
- greedy decoder = HF transformers' default short-form Whisper decoding (prompt <|startoftranscript|><|en|>
  <|transcribe|><|notimestamps|>, suppress_tokens, begin_suppress_tokens, greedy, max 448 positions), also returning
  the openai-whisper statistics (no-speech prob, avg log-prob, compression ratio) and optionally capturing the
  last decoder layer's MLP input (for the gate)
- openai threshold rule, Silero VAD front-end (faster-whisper defaults), Whisper's EnglishTextNormalizer, metrics
"""
import json
import math
import os
import zlib
from pathlib import Path

import numpy as np
import torch

WP = Path(__file__).resolve().parent.parent                      # weight-patching/
DATA = Path(os.environ.get("WHISPER_SILENCE_DATA", WP.parent / "data_audio"))
DERIVED = DATA / "derived"
SR = 16000
MAX_POSITIONS = 448
CATEGORIES = ("silence", "noise", "environmental", "music")
HELDOUT_SUITES = ("nonspeech", "padded", "wer", "bg", "other")
GUARD_SUITES = ("padded", "wer", "bg", "other")
PRACTICE_SUITES = ("practice_nonspeech", "practice_speech", "practice_padded", "practice_bg")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------------------------- data
def load_suite(name):
    """(manifest list of dicts, list of float32 arrays as views into a memory map)."""
    man = json.loads((DERIVED / f"{name}.json").read_text(encoding="utf-8"))
    buf = np.load(DERIVED / f"{name}.npy", mmap_mode="r")
    return man, [buf[m["offset"]:m["offset"] + m["n"]] for m in man]


# ----------------------------------------------------------------------------------------------- model
def gpu_setup():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)
        torch.backends.cuda.matmul.allow_tf32 = False      # fp32 exact, so tiny edits are not rounded away


def load(model_name):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(model_name, revision="973afd24965f72e36ca33b3055d56a652f456b4d")
    m = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32, attn_implementation="sdpa", revision="973afd24965f72e36ca33b3055d56a652f456b4d", use_safetensors=True)
    m.eval()
    return proc, m


@torch.no_grad()
def apply_edit(model, edit):
    """Copy {param name: full new tensor} into the model; returns {name: old tensor} (so it can be undone)."""
    params = dict(model.named_parameters())
    old = {}
    for k, t in edit.items():
        p = params[k]
        old[k] = p.detach().cpu().clone()
        p.data.copy_(t.to(p.device, p.dtype))
    return old


def count_changed(edit, model):
    params = dict(model.named_parameters())
    return int(sum(int((t.float() != params[k].detach().cpu().float()).sum()) for k, t in edit.items()))


def save_edit(edit, path, meta):
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().float().contiguous() for k, v in edit.items()}, str(path))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))


def load_edit(path):
    from safetensors.torch import load_file
    return load_file(str(path))


class Tok:
    """Special token ids and HF default decoding config (generation_config.json)."""

    def __init__(self, model, proc):
        gc, tk = model.generation_config, proc.tokenizer
        self.tokenizer = tk
        self.sot, self.eot = gc.decoder_start_token_id, gc.eos_token_id
        self.lang = gc.lang_to_id["<|en|>"]
        self.transcribe = gc.task_to_id["transcribe"]
        self.notimestamps = gc.no_timestamps_token_id
        ns = tk.convert_tokens_to_ids("<|nospeech|>")
        self.nospeech = ns if ns is not None and ns != tk.unk_token_id else tk.convert_tokens_to_ids("<|nocaptions|>")
        self.prompt = [self.sot, self.lang, self.transcribe, self.notimestamps]
        self.suppress = list(gc.suppress_tokens or [])
        self.begin_suppress = list(gc.begin_suppress_tokens or [])
        self.max_new = MAX_POSITIONS - len(self.prompt)

    def text(self, ids):
        return self.tokenizer.decode([i for i in ids if i < self.eot], skip_special_tokens=True)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)


def features(proc, audios, device=DEV):
    f = proc.feature_extractor([np.asarray(a, dtype=np.float32) for a in audios], sampling_rate=SR,
                               return_tensors="pt").input_features
    return f.to(device, torch.float32)


def compression_ratio(text):
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b)) if b else 0.0


@torch.no_grad()
def encode(model, feats):
    return model.model.encoder(feats).last_hidden_state


@torch.no_grad()
def greedy_decode(model, tok, enc, max_new=None, capture=False):
    """Greedy decoding from encoder states enc [B, T, d]; matches HF's default short-form generate.
    Returns per clip: tokens (without EOT), nsp (no-speech prob at the SOT position, raw softmax as openai-whisper),
    avg_logprob (openai-whisper), hit_max, top_first (top-10 first-token ids after HF suppression); with capture=True
    also x = last decoder layer's final_layer_norm output (= MLP input) at positions [0, 4 + n_tokens)."""
    B, dev = enc.shape[0], enc.device
    max_new = tok.max_new if max_new is None else min(max_new, tok.max_new)
    cap, h = [], None
    if capture:
        h = model.model.decoder.layers[-1].final_layer_norm.register_forward_hook(
            lambda m, i, o: cap.append(o.float().cpu()))
    try:
        ids = torch.tensor([tok.prompt] * B, device=dev)
        sup = torch.tensor(tok.suppress, device=dev, dtype=torch.long)
        bsup = torch.tensor(tok.begin_suppress, device=dev, dtype=torch.long)
        out = model(encoder_outputs=(enc,), decoder_input_ids=ids, use_cache=True)
        nsp = out.logits[:, 0].float().softmax(-1)[:, tok.nospeech].cpu()
        logits, past = out.logits[:, -1].float(), out.past_key_values
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        sum_lp, n_lp = torch.zeros(B, device=dev), torch.zeros(B, device=dev)
        toks, top_first = [], None
        for step in range(max_new):
            logits = logits.clone()
            if len(sup):
                logits[:, sup] = -math.inf
            if step == 0:
                if len(bsup):
                    logits[:, bsup] = -math.inf
                top_first = logits.topk(10, -1).indices.cpu()
            lp = logits.log_softmax(-1)
            nxt = torch.where(done, torch.full_like(logits[:, 0], tok.eot, dtype=torch.long), logits.argmax(-1))
            chosen = lp.gather(1, nxt[:, None]).squeeze(1)
            sum_lp += torch.where(done, torch.zeros_like(chosen), chosen)
            n_lp += (~done).float()
            toks.append(nxt)
            done = done | (nxt == tok.eot)
            if bool(done.all()):
                break
            out = model(encoder_outputs=(enc,), decoder_input_ids=nxt[:, None], past_key_values=past, use_cache=True)
            logits, past = out.logits[:, -1].float(), out.past_key_values
    finally:
        if h is not None:
            h.remove()
    T = torch.stack(toks, 1).cpu()
    X = torch.cat(cap, 1) if capture else None
    res = []
    for b in range(B):
        row = T[b].tolist()
        gen = row[:row.index(tok.eot)] if tok.eot in row else row
        r = dict(tokens=gen, nsp=float(nsp[b]), avg_logprob=float(sum_lp[b] / max(n_lp[b], 1)),
                 hit_max=tok.eot not in row, top_first=top_first[b].tolist())
        if capture:
            r["x"] = X[b, :min(4 + len(gen), X.shape[1])].clone()
        res.append(r)
    return res


def transcribe(model, proc, tok, audios, batch=32, capture=False, max_new=None):
    """Model must already be on DEV. None entries (VAD found no speech) give empty results."""
    res = [None] * len(audios)
    idx = [i for i, a in enumerate(audios) if a is not None]
    for k in range(0, len(idx), batch):
        part = idx[k:k + batch]
        enc = encode(model, features(proc, [audios[i] for i in part], next(model.parameters()).device))
        for i, r in zip(part, greedy_decode(model, tok, enc, capture=capture, max_new=max_new)):
            r["text"] = tok.text(r["tokens"])
            r["compression_ratio"] = compression_ratio(r["text"])
            res[i] = r
    return res


# ----------------------------------------------------------------------------------------------- thresholds
OPENAI_DEFAULT = dict(no_speech=0.6, logprob=-1.0)
# tuned on PRACTICE data only in the original run (whisper_halluc/tune_thresholds.py -> runs/whisper/thresholds.json):
# lowest practice hallucination with practice clean WER <= +0.3, practice bg WER <= +1 and <= +1 pt dropped
PRACTICE_TUNED = dict(no_speech=0.2, logprob=None)


def threshold_keep(r, cfg):
    """openai-whisper's short-form rule on one greedy pass: skip if nsp > no_speech and avg_logprob <= logprob."""
    ns, lp = cfg.get("no_speech"), cfg.get("logprob")
    return not (ns is not None and r["nsp"] > ns and (lp is None or r["avg_logprob"] <= lp))


# ----------------------------------------------------------------------------------------------- VAD
VAD_OPTS = dict(threshold=0.5, min_speech_duration_ms=0, min_silence_duration_ms=2000, speech_pad_ms=400)
_vad = None


def vad_segments(audio):
    """Silero VAD (CPU) with faster-whisper's vad_filter defaults -> [[start, end], ...] in samples."""
    global _vad
    from silero_vad import get_speech_timestamps, load_silero_vad
    if _vad is None:
        import warnings
        warnings.filterwarnings("ignore", message=".*torch.jit.load.*")
        _vad = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(np.array(audio, dtype=np.float32)), _vad, sampling_rate=SR, **VAD_OPTS)
    return [[int(t["start"]), int(t["end"])] for t in ts]


def vad_collect(audio, segs):
    """faster-whisper style: concatenate the detected speech regions (None = no speech)."""
    if not segs:
        return None
    return np.concatenate([np.asarray(audio[s:e], dtype=np.float32) for s, e in segs])


# ----------------------------------------------------------------------------------------------- metrics
_norm = None


def norm(text):
    """Whisper's EnglishTextNormalizer (removes punctuation, symbols, [..], (..)); whitespace-collapsed."""
    global _norm
    if _norm is None:
        from huggingface_hub import hf_hub_download
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        try:
            spell = json.loads(Path(hf_hub_download("openai/whisper-small", "normalizer.json", revision="973afd24965f72e36ca33b3055d56a652f456b4d")).read_text("utf-8"))
        except Exception:
            spell = {}
        _norm = EnglishTextNormalizer(spell)
    return " ".join(_norm(text or "").split())


SOUND_TAG_WORDS = {"music", "applause", "laughter", "laughing", "laughs", "silence", "noise", "static", "inaudible",
                   "cheering", "instrumental", "upbeat", "playing", "background", "soft", "dramatic", "gentle",
                   "sighs", "coughs", "coughing", "clapping"}


def is_sound_tag(n):
    w = n.split()
    return 0 < len(w) <= 4 and all(x in SOUND_TAG_WORDS for x in w)


def word_stats(refs, hyps):
    """Corpus WER (%), insertions per reference word (%), share with any insertion, share fully dropped."""
    import jiwer
    S = dict(sub=0, dele=0, ins=0, nref=0)
    anyins, dropped = [], []
    for r, h in zip(refs, hyps):
        rn, hn = norm(r), norm(h)
        if not rn:
            continue
        o = jiwer.process_words(rn, hn)
        S["sub"] += o.substitutions; S["dele"] += o.deletions; S["ins"] += o.insertions; S["nref"] += len(rn.split())
        anyins.append(o.insertions > 0)
        dropped.append(not hn)
    n = max(S["nref"], 1)
    return dict(wer=round(100 * (S["sub"] + S["dele"] + S["ins"]) / n, 3), ins=round(100 * S["ins"] / n, 3),
                any_ins=round(100 * float(np.mean(anyins)), 2) if anyins else 0.0,
                dropped=round(100 * float(np.mean(dropped)), 2) if dropped else 0.0, n=len(anyins))
