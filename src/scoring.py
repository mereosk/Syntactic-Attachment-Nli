"""Scoring premise/hypothesis pairs with NLI classifiers and generative models."""
import gc
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
from tqdm.auto import tqdm
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForCausalLM, AutoModelForSeq2SeqLM)

from .config import (API_MODELS, OPENROUTER_URL, OPENAI_URL, API_CACHE_PATH,
                     OPENROUTER_API_KEY, OPENAI_API_KEY, DEVICE, FAMILY)
from .prompts import (LABELS, ANSWER_STRINGS, VARIANTS, build_prompt, norm_answer)


def _meta(p):
    return {k: p[k] for k in ("item_id", "section", "category", "slot")}


#  NLI classifiers 
def classifier_label_indices(config):
    idx = {}
    for i, lab in config.id2label.items():
        l = lab.lower()
        if l.startswith("not"):
            continue
        if "entail" in l:       idx["entailment"] = int(i)
        elif "neutral" in l:    idx["neutral"] = int(i)
        elif "contradict" in l: idx["contradiction"] = int(i)
    missing = set(LABELS) - set(idx)
    assert not missing, f"Label(s) {missing} not found in {config.id2label}"
    return idx


@torch.no_grad()
def score_classifier(model, tok, pairs, batch_size=32):
    idx = classifier_label_indices(model.config)
    rows = []
    for s in tqdm(range(0, len(pairs), batch_size), desc="classifier"):
        batch = pairs[s:s + batch_size]
        enc = tok([p["premise"] for p in batch], [p["hypothesis"] for p in batch],
                  padding=True, truncation=True, return_tensors="pt").to(model.device)
        probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
        for p, pr in zip(batch, probs):
            rows.append({**_meta(p), "template": "native", "order": "native",
                         **{f"p_{lab}": float(pr[idx[lab]]) for lab in LABELS},
                         "coverage": 1.0, "n_missing": 0})
    return rows


#  Local generative models 
def answer_token_ids(tok):
    """All vocabulary tokens that are a surface variant of an answer
    ('A', ' A', '\nA', 'A.', 'A)', ...). Their probabilities are summed per answer."""
    ids = {w: [] for w in ANSWER_STRINGS}
    for t in range(len(tok)):
        n = norm_answer(tok.decode([t]))
        if n in ids:
            ids[n].append(t)
    for w, v in ids.items():
        if not v:
            first = tok.encode(w, add_special_tokens=False)[0]
            print(f"  WARNING: no single-token form of '{w}'; using first sub-token {tok.decode([first])!r}")
            v.append(first)
    print("  answer tokens:", {w: [tok.decode([i]) for i in v][:10] for w, v in ids.items()})
    return ids


def show_top_tokens(tok, logp_row, k=15):
    top = torch.topk(logp_row, k)
    return [(repr(tok.decode([int(i)])), round(float(v.exp()), 3)) for v, i in zip(top.values, top.indices)]


def format_for_model(tok, prompt, kind):
    if kind == "seq2seq" or not getattr(tok, "chat_template", None):
        return prompt
    extra = {"enable_thinking": False} if "enable_thinking" in tok.chat_template else {}
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True, **extra)


@torch.no_grad()
def score_generative_local(model, tok, kind, pairs, batch_size=8):
    ans_ids = answer_token_ids(tok)
    jobs = [(p, t, o) for p in pairs for (t, o) in VARIANTS]
    rows = []
    for s in tqdm(range(0, len(jobs), batch_size), desc=kind):
        batch = jobs[s:s + batch_size]
        built = [build_prompt(t, o, p["premise"], p["hypothesis"]) for p, t, o in batch]
        texts = [format_for_model(tok, txt, kind) for txt, _ in built]
        if kind == "seq2seq":
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            dec = torch.full((len(batch), 1), model.config.decoder_start_token_id,
                             dtype=torch.long, device=model.device)
            logits = model(**enc, decoder_input_ids=dec).logits[:, -1, :]
        else:
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            logits = model(**enc).logits[:, -1, :]
        logp = torch.log_softmax(logits.float(), dim=-1)
        if s == 0:                                   # inspect what the model actually wants to output
            for (p, t, o), row in list(zip(batch, logp))[:3]:
                print(f"  top tokens [{p['item_id']} {p['slot']} {t}-{o}]:", show_top_tokens(tok, row))
        for (p, t, o), (_, amap), row in zip(batch, built, logp):
            ans = {w: row[ids].exp().sum().item() for w, ids in ans_ids.items()}
            probs = {amap[w]: v for w, v in ans.items()}
            cov = sum(probs.values())
            rows.append({**_meta(p), "template": t, "order": o,
                         **{f"p_{lab}": probs[lab] / cov for lab in LABELS},
                         "coverage": cov, "n_missing": 0})
    return rows


#  API models (top-20 logprobs) 
api_cache = json.load(open(API_CACHE_PATH)) if os.path.exists(API_CACHE_PATH) else {}
_cache_lock = threading.Lock()


def _save_api_cache():
    with _cache_lock:
        snapshot = dict(api_cache)
    with open(API_CACHE_PATH, "w") as f:
        json.dump(snapshot, f)


def _request(slug, prompt):
    cfg = API_MODELS[slug]
    if cfg["backend"] == "openrouter":
        key = hashlib.sha1(json.dumps([slug, prompt, cfg["provider"]]).encode()).hexdigest()
        url, auth = OPENROUTER_URL, OPENROUTER_API_KEY
        payload = {"model": slug, "provider": cfg["provider"], **cfg.get("extra", {})}
    else:
        key = hashlib.sha1(json.dumps(["openai-direct", slug, prompt]).encode()).hexdigest()
        url, auth = OPENAI_URL, OPENAI_API_KEY
        payload = {"model": slug, **cfg.get("extra", {})}
    payload.update({"messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1, "temperature": 0,
                    "logprobs": True, "top_logprobs": 20})
    return key, url, auth, payload


def api_answer_probs(slug, prompt, max_retries=6):
    key, url, auth, payload = _request(slug, prompt)
    with _cache_lock:
        if key in api_cache:
            return api_cache[key]
    headers = {"Authorization": f"Bearer {auth}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            resp = r.json()
            top = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
            continue
        probs, found = {w: 0.0 for w in ANSWER_STRINGS}, set()
        for e in top:
            n = norm_answer(e["token"])
            if n in probs:
                probs[n] += float(np.exp(e["logprob"]))
                found.add(n)
        missing = [w for w in ANSWER_STRINGS if w not in found]
        coverage = sum(probs[w] for w in found)
        floor = float(np.exp(min(e["logprob"] for e in top)))   # upper bound for unseen answers
        for w in missing:
            probs[w] = floor
        result = {"probs": probs, "coverage": coverage, "missing": missing,
                  "served_by": resp.get("provider") or resp.get("model"),
                  "top": [(e["token"], round(float(np.exp(e["logprob"])), 3)) for e in top[:10]]}
        with _cache_lock:
            api_cache[key] = result
        return result
    raise RuntimeError(f"{slug} failed after {max_retries} tries: {last_err}")


def score_generative_api(slug, pairs):
    jobs = [(p, t, o) for p in pairs for (t, o) in VARIANTS]
    built = [build_prompt(t, o, p["premise"], p["hypothesis"]) for p, t, o in jobs]
    results = []
    with ThreadPoolExecutor(max_workers=API_MODELS[slug].get("workers", 4)) as ex:
        for i, res in enumerate(tqdm(ex.map(lambda b: api_answer_probs(slug, b[0]), built),
                                     total=len(built), desc=slug)):
            results.append(res)
            if i % 200 == 199:
                _save_api_cache()           # periodic save: a crash loses at most 200 calls
    _save_api_cache()
    rows, served = [], set()
    for (p, t, o), (_, amap), res in zip(jobs, built, results):
        probs = {amap[w]: v for w, v in res["probs"].items()}
        z = sum(probs.values())
        served.add(res["served_by"])
        rows.append({**_meta(p), "template": t, "order": o,
                     **{f"p_{lab}": probs[lab] / z for lab in LABELS},
                     "coverage": res["coverage"], "n_missing": len(res["missing"])})
    print("  served by:", served)
    return rows


def run_one(kind, name, short, pairs_):
    model = tok = None
    try:
        if kind == "classifier":
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE).eval()
            rows = score_classifier(model, tok, pairs_)
        elif kind == "seq2seq":
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float32).to(DEVICE).eval()
            rows = score_generative_local(model, tok, kind, pairs_)
        elif kind == "causal":
            tok = AutoTokenizer.from_pretrained(name)
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float16, device_map="auto").eval()
            rows = score_generative_local(model, tok, kind, pairs_)
        elif kind == "api":
            rows = score_generative_api(name, pairs_)
        else:
            raise ValueError(kind)
    finally:
        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for r in rows:
        r["model"], r["family"] = short, FAMILY[short]
    return rows
