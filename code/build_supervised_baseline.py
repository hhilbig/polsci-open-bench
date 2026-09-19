#!/usr/bin/env python3
"""
Supervised baselines for the benchmark tasks, as a function of training-set size.

The benchmark asks which LLM to use for prompt-based classification. The obvious
question it does not answer is why not train a classifier instead. That question
has a known answer in the abstract (with enough labels, supervised methods win)
and an unknown answer in practice: how many hand-coded labels does it take? That
is the cost the LLM lets a researcher avoid, so the crossover point is the
decision-relevant quantity.

Design. Each task samples 500 items from a larger frame, so the unsampled
remainder is free labelled training data and the test set can be the exact items
every LLM was scored on. Training and test text are rendered through the same
`task_registry` template the LLM runner uses, so the classifier sees byte-identical
input. Scores use `scoring.headline_f1` with the same support-only rule and the
same pinned label set as `output/summary.csv`, so supervised and LLM numbers are
directly comparable and the existing paired-by-item bootstrap applies.

Two methods:
  tfidf  -- TF-IDF + logistic regression, the floor. If frozen embeddings do not
            beat bag-of-words, they are not earning their cost.
  e5     -- frozen intfloat/multilingual-e5-large + logistic regression. Six of
            the tasks are not English, which rules out English-only encoders.

Usage:
  python3 code/build_supervised_baseline.py --method tfidf
  python3 code/build_supervised_baseline.py --embed-only      # cache embeddings
  python3 code/build_supervised_baseline.py --method e5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

import task_registry as tr
from scoring import headline_f1, scored_labels
from task_registry import load_task_definitions

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "output"
EMB_DIR = OUT / "embeddings"

EMBED_MODEL = "intfloat/multilingual-e5-large"
EMBED_DIM = 1024
EMBED_PREFIX = "query: "          # E5 requires a prefix; symmetric tasks use query:
EMBED_MAX_TOKENS = 512            # E5's context limit; see the caveat in the report

TRAIN_SIZES = [50, 100, 250, 500, 1000, 2000]
# Only this many pool rows per task are embedded and drawn from. Two reasons.
# The curve's purpose is the region a researcher might actually hand-code, and
# nobody hand-codes 36,838 examples, so the full-remainder point answered a
# question no one asks. It was also the expensive one: a TF-IDF bigram fit over
# 36,838 documents inside a 3x3 grid search dominated the whole run, and the
# full-frame embedding cache would have needed 505 MB on a disk that had 118 MB
# free. Capping keeps every training size genuinely comparable across tasks.
POOL_CAP = 3000
SEEDS = [0, 1, 2, 3, 4]
C_GRID = [0.1, 1.0, 10.0]
CV_MIN_N = 100                    # below this, cross-validating C is unstable


# --------------------------------------------------------------------------
# Split reconstruction
# --------------------------------------------------------------------------

def task_split(task):
    """Recover the benchmark's own test sample plus the unused training pool.

    The frame is transformed exactly as the loader transforms it (label_map then
    exclude_labels) before indices are drawn, otherwise the recovered indices do
    not line up with what the LLM actually saw.
    """
    spec = yaml.safe_load(Path(task["manifest_path"]).read_text())
    gt_spec = spec["ground_truth"]
    frame = pd.read_csv(task["data_path"], low_memory=False)

    label_map = tr._label_map(gt_spec)
    excluded = tr._excluded_labels(gt_spec)
    if label_map or excluded:
        column = gt_spec["column"]
        if label_map:
            frame[column] = frame[column].astype(str).replace(label_map)
        if excluded:
            frame = frame[~frame[column].astype(str).isin(excluded)].reset_index(drop=True)

    v1, v2 = tr._v1_v2_indices(len(frame), **task["sampling"])
    test_idx = list(v1) + list(v2)
    pool_idx = sorted(set(range(len(frame))) - set(test_idx))

    texts = [tr._render_user_content(frame.iloc[i], spec["text"]) for i in range(len(frame))]
    return frame, np.array(texts, dtype=object), np.array(test_idx), np.array(pool_idx), gt_spec


def capped_pool(pool_idx):
    """The training rows both methods draw from.

    Capped at POOL_CAP and chosen deterministically, so TF-IDF and E5 train on
    identical draws and the two curves differ only by representation. Returned in
    frame order.
    """
    if len(pool_idx) <= POOL_CAP:
        return np.asarray(pool_idx)
    rng = np.random.default_rng(20260918)
    return np.sort(rng.choice(np.asarray(pool_idx), POOL_CAP, replace=False))


def embed_row_indices(test_idx, pool_idx):
    """Frame rows that need an embedding: the test set plus the capped pool."""
    return np.sort(np.unique(np.concatenate([np.asarray(test_idx), capped_pool(pool_idx)])))


def gold_matrix(task, frame, gt_spec, idx):
    """Gold labels for the given rows, in the shape the scorer expects."""
    kind = task["label_kind"]
    if kind == "multi_binary":
        columns = gt_spec.get("columns") or {l: f"gt_{l}" for l in task["labels"]}
        return np.column_stack(
            [frame.iloc[idx][columns[l]].astype(int).values for l in task["labels"]]
        )
    values = frame.iloc[idx][gt_spec["column"]]
    if kind == "binary":
        return values.astype(int).values
    return values.astype(str).values


def scoring_frame(task, gold, pred):
    """Assemble the gt_/pred_ columns `scoring.headline_f1` reads."""
    kind = task["label_kind"]
    if kind == "multi_binary":
        data = {}
        for j, label in enumerate(task["labels"]):
            data[f"gt_{label}"] = gold[:, j]
            data[f"pred_{label}"] = pred[:, j]
        return pd.DataFrame(data)
    key = task["label_key"]
    return pd.DataFrame({f"gt_{key}": gold, f"pred_{key}": pred})


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

def embed_texts(texts, batch_size=64, device=None):
    """Mean-pooled, L2-normalised E5 embeddings."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL).to(device).eval()

    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float16)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = [EMBED_PREFIX + str(t) for t in texts[start:start + batch_size]]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=EMBED_MAX_TOKENS, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out[start:start + len(chunk)] = pooled.cpu().numpy().astype(np.float16)
    return out


def cached_embeddings(task_name, texts, rows, force=False):
    """Embeddings for `rows` of the frame, cached with the row index they cover.

    A cache covering the whole frame is a valid superset of any row subset, so
    earlier full-frame caches stay usable rather than being recomputed.
    """
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    npy = EMB_DIR / f"{task_name}.f16.npy"
    idx_path = EMB_DIR / f"{task_name}.rows.npy"
    meta = EMB_DIR / f"{task_name}.json"
    if npy.exists() and meta.exists() and not force:
        info = json.loads(meta.read_text())
        arr = np.load(npy)
        covered = np.load(idx_path) if idx_path.exists() else np.arange(arr.shape[0])
        if (info.get("model") == EMBED_MODEL and arr.shape[1] == EMBED_DIM
                and np.isin(rows, covered).all()):
            return arr, covered
        print(f"  [{task_name}] cache does not cover the needed rows, recomputing")
    arr = embed_texts([texts[i] for i in rows])
    np.save(npy, arr)
    np.save(idx_path, np.asarray(rows))
    meta.write_text(json.dumps({
        "model": EMBED_MODEL, "dim": EMBED_DIM, "pooling": "mean",
        "normalized": True, "prefix": EMBED_PREFIX,
        "max_tokens": EMBED_MAX_TOKENS, "n_texts": int(len(rows)),
        "n_frame": int(len(texts)),
    }, indent=2))
    return arr, np.asarray(rows)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def fit_predict(kind, X_train, y_train, X_test):
    """Logistic regression, class-weighted, with C tuned when n allows."""
    n = len(y_train)
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    if n >= CV_MIN_N:
        # Cross-validating on the training subset only; no test leakage.
        folds = min(3, int(np.min(np.bincount(
            y_train if kind == "binary" else pd.factorize(y_train)[0]))) ) if kind != "multi_binary" else 3
        folds = max(2, min(3, folds))
        try:
            search = GridSearchCV(base, {"C": C_GRID}, cv=folds, scoring="f1_macro", n_jobs=1)
            if kind == "multi_binary":
                model = OneVsRestClassifier(search)
            else:
                model = search
            model.fit(X_train, y_train)
            return model.predict(X_test)
        except ValueError:
            pass  # too few members in some class for CV; fall through to fixed C
    model = OneVsRestClassifier(base) if kind == "multi_binary" else base
    model.fit(X_train, y_train)
    return model.predict(X_test)


def draw_indices(pool_idx, gold_pool, kind, n, rng):
    """Sample n training rows, keeping at least one of each observed class."""
    if n >= len(pool_idx):
        return np.arange(len(pool_idx))
    if kind == "multi_binary":
        return rng.choice(len(pool_idx), n, replace=False)
    classes, first = np.unique(gold_pool, return_index=True)
    if len(classes) > n:
        return rng.choice(len(pool_idx), n, replace=False)
    rest = np.setdiff1d(np.arange(len(pool_idx)), first)
    extra = rng.choice(rest, n - len(first), replace=False)
    return np.concatenate([first, extra])


def run_task(task, method, embeddings=None, covered=None):
    frame, texts, test_idx, full_pool, gt_spec = task_split(task)
    if len(full_pool) == 0:
        return []
    pool_idx = capped_pool(full_pool)

    gold_test = gold_matrix(task, frame, gt_spec, test_idx)
    gold_pool = gold_matrix(task, frame, gt_spec, pool_idx)
    kind = task["label_kind"]

    # Pin the scored label set from the test gold, matching how summary.csv is built.
    pinned = scored_labels(task, scoring_frame(task, gold_test, gold_test), support_only=True)

    if method == "e5":
        # `embeddings` holds only the rows in `covered`; map frame index -> position.
        position = {int(r): i for i, r in enumerate(covered)}
        X = embeddings.astype(np.float32)
        X_test = X[[position[int(i)] for i in test_idx]]
        pool_features = X[[position[int(i)] for i in pool_idx]]
    else:
        X_test = pool_features = None  # built per draw, since TF-IDF fits on the training text

    rows = []
    for n in TRAIN_SIZES:
        is_full = False
        if n > len(pool_idx):
            rows.append(dict(task=task["name"], method=method, n_train=n, seed=np.nan,
                             headline_f1=np.nan, accuracy=np.nan, n_train_actual=np.nan,
                             note="remainder smaller than requested size"))
            continue
        for seed in SEEDS:
            rng = np.random.default_rng(20260918 + seed)
            sel = draw_indices(pool_idx, gold_pool, kind, n, rng)
            y_train = gold_pool[sel]

            if method == "tfidf":
                vec = TfidfVectorizer(sublinear_tf=True, min_df=1, ngram_range=(1, 2),
                                      max_features=200_000, analyzer="word")
                train_texts = texts[pool_idx][sel]
                try:
                    Xtr = vec.fit_transform(train_texts)
                except ValueError:
                    continue
                Xte = vec.transform(texts[test_idx])
            else:
                Xtr = pool_features[sel]
                Xte = X_test

            pred = fit_predict(kind, Xtr, y_train, Xte)
            sf = scoring_frame(task, gold_test, pred)
            f1 = headline_f1(task, sf, support_only=True, label_subset=pinned)
            if kind == "multi_binary":
                acc = float((gold_test == pred).all(axis=1).mean())
            else:
                acc = float((gold_test == pred).mean())
            rows.append(dict(task=task["name"], method=method, n_train=n, seed=seed,
                             headline_f1=f1, accuracy=acc,
                             n_train_actual=int(len(sel)), note=""))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["tfidf", "e5"], default="tfidf")
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--force-embed", action="store_true")
    ap.add_argument("--only-task")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    tasks = [t for t in load_task_definitions(active_only=True)
             if not args.only_task or t["name"] == args.only_task]

    if args.embed_only or args.method == "e5":
        for task in tasks:
            _, texts, test_idx, pool_idx, _ = task_split(task)
            if len(pool_idx) == 0:
                print(f"[skip] {task['name']}: no training remainder")
                continue
            rows = embed_row_indices(test_idx, pool_idx)
            arr, _ = cached_embeddings(task["name"], texts, rows, force=args.force_embed)
            print(f"[embed] {task['name']:38s} {arr.shape} of {len(texts)} frame rows")
        if args.embed_only:
            return

    rows = []
    for task in tasks:
        _, texts, _, pool_idx, _ = task_split(task)
        if len(pool_idx) == 0:
            print(f"[skip] {task['name']}: no training remainder")
            continue
        emb = cov = None
        if args.method == "e5":
            # NB: not `rows` -- that name is the result accumulator below.
            embed_rows = embed_row_indices(task_split(task)[2], pool_idx)
            emb, cov = cached_embeddings(task["name"], texts, embed_rows)
        got = run_task(task, args.method, embeddings=emb, covered=cov)
        rows.extend(got)
        done = [r for r in got if not np.isnan(r["headline_f1"])]
        if done:
            best = max(done, key=lambda r: r["headline_f1"])
            print(f"[{args.method}] {task['name']:38s} best F1 {best['headline_f1']:.3f} "
                  f"at n={best['n_train']}")

    out = Path(args.output) if args.output else OUT / f"supervised_baseline_{args.method}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
