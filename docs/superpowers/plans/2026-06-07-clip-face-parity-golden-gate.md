# CLIP & Face Parity Golden-Reference Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an automated, weights-gated parity gate that fails when MLX embeddings drift below threshold versus reference embeddings frozen from the literal upstream ONNX models Immich ships, with committed fixtures and golden artifacts.

**Architecture:** Approach C (hybrid). Add a literal-ONNX reference backend to the existing diagnostic scripts (`clip_parity.py`, `embedding_parity.py`) and factor the face script's already-present upstream-ONNX pass. A thin run-once generator (`scripts/gen_parity_golden.py`) calls those backends on committed fixtures and writes `.npz` + JSON goldens. New weights-gated tests load the goldens and compare against the MLX production backends. Harness-only test files are renamed to `*_harness.py`.

**Tech Stack:** Python 3.11, MLX, onnxruntime, insightface, huggingface_hub, numpy, Pillow, pytest. All work is inside the `ml/` submodule.

---

## Working directory & conventions

- **All paths below are relative to the `ml/` submodule root:** `/Users/john/Code/immich-apple-silicon/.claude/worktrees/cheeky-jumping-hinton/ml`. Run every command from there unless stated otherwise.
- Use the venv interpreter: `.venv/bin/python` and `.venv/bin/python -m pytest`.
- This is the `ml` submodule (a separate fork). Commit **inside `ml`** for each task. The parent-repo pointer bump happens once at the end (Task 14), per the repo's git workflow.
- **No bead IDs** in any committed file, comment, or commit message (repo rule).
- Commit messages end with the `Co-Authored-By` trailer used elsewhere in this repo.
- Default `pytest` runs under the conftest's forced `STUB_MODE=true` and must stay hermetic. The new gated tests skip there; they are verified locally with `ML_RUN_PARITY=1` (see Task 13).

## File structure (created / modified)

| Path | Action | Responsibility |
|---|---|---|
| `tests/fixtures/README.md` | create | Provenance + license for every committed fixture/golden |
| `tests/fixtures/queries.txt` | create | Fixed CLIP text-query list |
| `tests/fixtures/clip/*.jpg` | create | ~8 public-domain CLIP photos (≤512px) |
| `tests/fixtures/faces/<id>/*.jpg` | create | LFW subset (~3 ids × 2 imgs) |
| `tests/fixtures/golden/{openai_clip,siglip2,face}.npz` | create | Frozen ONNX reference embeddings + metadata |
| `tests/fixtures/golden/{openai_clip,siglip2,face}.json` | create | Human-readable manifests |
| `scripts/fetch_pd_clip_fixtures.py` | create | Download + resize the PD CLIP photos (run-once) |
| `scripts/clip_parity.py` | modify | Add `embed_onnx()` OpenAI-CLIP ONNX backend + `--ref onnx` |
| `scripts/embedding_parity.py` | modify | Add `embed_onnx_siglip2()` SigLIP2 ONNX backend + `--ref onnx` |
| `scripts/face_embedding_parity.py` | modify | Factor `upstream_embeddings()` so the generator can freeze golden |
| `scripts/gen_parity_golden.py` | create | Run-once generator → writes golden `.npz` + `.json` |
| `tests/_parity_gate.py` | create | Auto-detect/skip vs `ML_RUN_PARITY=1` hard-fail helper |
| `tests/test_parity_gate.py` | create | Hermetic unit test for the gate helper |
| `tests/test_clip_golden_parity.py` | create | Weights-gated CLIP gate (SigLIP2 + OpenAI-CLIP) |
| `tests/test_face_golden_parity.py` | create | Weights-gated face gate |
| `tests/test_embedding_parity.py` → `tests/test_embedding_parity_harness.py` | rename | Harness math only |
| `tests/test_face_embedding_parity.py` → `tests/test_face_embedding_parity_harness.py` | rename | Harness math only |
| `README.md` | modify | Document the automated parity gate |

---

## Task 1: Committed text queries + fixtures README scaffold

**Files:**
- Create: `tests/fixtures/queries.txt`
- Create: `tests/fixtures/README.md`

- [ ] **Step 1: Write the query list**

Reuse the existing `DEFAULT_QUERIES` content from `scripts/embedding_parity.py` (lines ~77–92) so the gate exercises the same queries the manual harness uses. Read that list and copy each query, one per line, into `tests/fixtures/queries.txt`. Example shape (use the actual list from the script, not these placeholders):

```
a photo of a dog
a city skyline at night
...
```

- [ ] **Step 2: Create the fixtures README scaffold**

```markdown
# Parity test fixtures

Committed inputs and golden references for the weights-gated parity gate
(`tests/test_clip_golden_parity.py`, `tests/test_face_golden_parity.py`).

## queries.txt
Fixed CLIP text queries. Mirrors `scripts/embedding_parity.py::DEFAULT_QUERIES`.

## clip/
Public-domain photos for CLIP image parity. Provenance per file:

| file | source URL | license |
|------|------------|---------|
| (filled by scripts/fetch_pd_clip_fixtures.py) | | |

## faces/
LFW subset for face parity (identity subdirectories, ≥2 images each).
Source: `logasja/lfw` (HuggingFace dataset). LFW is a research-use face
verification benchmark; only a minimal subset is committed here.

## golden/
Reference embeddings frozen from the upstream ONNX models Immich ships
(`immich-app/*` repos, insightface `buffalo_l`). Regenerate with
`scripts/gen_parity_golden.py`. Each `.json` manifest records the ONNX repo,
resolved commit SHA, onnxruntime version, dim, and generation date.
```

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/queries.txt tests/fixtures/README.md
git commit -m "test(parity): add committed query list + fixtures README scaffold

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: CLIP public-domain fixture images

**Files:**
- Create: `scripts/fetch_pd_clip_fixtures.py`
- Create: `tests/fixtures/clip/*.jpg` (output)
- Modify: `tests/fixtures/README.md` (provenance table)

NASA imagery is explicitly public domain and served from stable
`images-assets.nasa.gov` URLs — a clean, reproducible PD source. The script
downloads a documented list, converts to RGB JPEG, resizes the long side to
≤512px, and appends provenance to the README.

- [ ] **Step 1: Write the fetch script**

```python
#!/usr/bin/env python3
"""Download public-domain CLIP fixture photos (run-once).

NASA images are public domain. Each entry below is (filename, source_url).
Verify each URL resolves and is PD before committing the output.
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "clip"
MAX_SIDE = 512

# Diverse public-domain content (NASA image library). Confirm each resolves.
SOURCES: list[tuple[str, str]] = [
    ("earth_blue_marble.jpg", "https://images-assets.nasa.gov/image/PIA18033/PIA18033~orig.jpg"),
    ("apollo17_moon.jpg", "https://images-assets.nasa.gov/image/as17-148-22727/as17-148-22727~orig.jpg"),
    ("jupiter_juno.jpg", "https://images-assets.nasa.gov/image/PIA21974/PIA21974~orig.jpg"),
    ("nebula_hubble.jpg", "https://images-assets.nasa.gov/image/GSFC_20171208_Archive_e000075/GSFC_20171208_Archive_e000075~orig.jpg"),
    ("astronaut_spacewalk.jpg", "https://images-assets.nasa.gov/image/iss040e090540/iss040e090540~orig.jpg"),
    ("shuttle_launch.jpg", "https://images-assets.nasa.gov/image/sts121-s-005/sts121-s-005~orig.jpg"),
    ("mars_curiosity.jpg", "https://images-assets.nasa.gov/image/PIA16239/PIA16239~orig.jpg"),
    ("aurora_iss.jpg", "https://images-assets.nasa.gov/image/iss030e187157/iss030e187157~orig.jpg"),
]


def fetch_one(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "parity-fixtures/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (documented PD URLs)
        return resp.read()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, url in SOURCES:
        print(f"[fetch] {name} <- {url}")
        img = Image.open(io.BytesIO(fetch_one(url))).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(OUT / name, format="JPEG", quality=90)
        rows.append((name, url))
    print(f"\n[done] wrote {len(rows)} images to {OUT}")
    print("\nProvenance rows (paste into tests/fixtures/README.md):")
    for name, url in rows:
        print(f"| {name} | {url} | NASA — public domain |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/fetch_pd_clip_fixtures.py`
Expected: 8 JPEGs written to `tests/fixtures/clip/`, plus a printed provenance table.
If any URL 404s, replace it with another NASA PD image and rerun. Confirm each file is a valid small JPEG: `.venv/bin/python -c "from PIL import Image,ImageFile; import glob; [print(p, Image.open(p).size) for p in glob.glob('tests/fixtures/clip/*.jpg')]"`

- [ ] **Step 3: Fill the README provenance table**

Paste the printed rows into the `clip/` table in `tests/fixtures/README.md`.

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_pd_clip_fixtures.py tests/fixtures/clip tests/fixtures/README.md
git commit -m "test(parity): add public-domain CLIP fixture photos + fetch script

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: ONNX reference backend for OpenAI-CLIP (`clip_parity.py`)

**Files:**
- Modify: `scripts/clip_parity.py`

Mirror the proven `OnnxBackend` in `scripts/clip_benchmark.py` (lines 208–264).
Add `ViT-B-32__openai` to a local repo map, run the `immich-app` visual+textual
ONNX with Immich's preprocessing, and expose it as a `--ref onnx` variant. The
text path uses `clean_text(canonicalize=False)` + the open_clip tokenizer to
match what the existing `immich` reference (`embed_openclip_refs`) feeds, so a
later sanity check (ONNX ≈ open_clip ≈ 1.0) confirms equivalence.

- [ ] **Step 1: Add the ONNX backend function**

Insert after `embed_openclip_refs` (around line 222). Note `clip_parity.py`
already imports `gc`, `io`, `np`, `Image`, and the `CLIP_*` constants.

```python
# Immich CLIP name -> upstream HF repo (ONNX export) + open_clip arch (tokenizer).
_ONNX_REPO = {
    "ViT-B-32__openai": ("immich-app/ViT-B-32__openai", "ViT-B-32"),
    "ViT-B-16__openai": ("immich-app/ViT-B-16__openai", "ViT-B-16-quickgelu"),
    "ViT-L-14__openai": ("immich-app/ViT-L-14__openai", "ViT-L-14-quickgelu"),
}


def embed_onnx(
    model_name: str,
    images: list[tuple[str, bytes]],
    queries: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Literal upstream ONNX export (the checkpoint the Immich index was built
    with) via onnxruntime, fed Immich's exact transform + tokenization.

    Returns (image_embeds [N,D], text_embeds [M,D]), L2-normalized float32.
    """
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    import open_clip

    from src.models.immich_preprocess import clean_text, siglip_image_pixels

    if model_name not in _ONNX_REPO:
        raise SystemExit(f"No ONNX repo mapping for {model_name!r}; add it to _ONNX_REPO.")
    repo, arch = _ONNX_REPO[model_name]
    vis_path = hf_hub_download(repo, "visual/model.onnx")
    txt_path = hf_hub_download(repo, "textual/model.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    vis = ort.InferenceSession(vis_path, so, providers=["CPUExecutionProvider"])
    txt = ort.InferenceSession(txt_path, so, providers=["CPUExecutionProvider"])
    vis_in = vis.get_inputs()[0].name
    txt_in = txt.get_inputs()[0].name
    tokenizer = open_clip.get_tokenizer(arch)

    def _l2(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return (v / n if n > 0 else v).astype(np.float32)

    img_out = []
    for _, b in images:
        pil = Image.open(io.BytesIO(b)).convert("RGB")
        px = siglip_image_pixels(pil, size=CLIP_IMAGE_SIZE, mean=CLIP_MEAN, std=CLIP_STD).astype(np.float32)
        img_out.append(_l2(vis.run(None, {vis_in: px})[0][0]))
    txt_out = []
    for q in queries:
        ids = tokenizer([clean_text(q, canonicalize=False)]).numpy().astype(np.int32)
        txt_out.append(_l2(txt.run(None, {txt_in: ids})[0][0]))
    del vis, txt
    gc.collect()
    return np.stack(img_out), np.stack(txt_out)
```

- [ ] **Step 2: Wire `onnx` into `--ref` and the report loop**

In `main()` (around line 232) add `"onnx"` to the `--ref` choices and the
default list, and ensure the report orders it. Change the choices line:

```python
    ap.add_argument(
        "--ref",
        nargs="+",
        choices=["immich", "openclip", "onnx"],
        default=["onnx", "immich", "openclip"],
        help="reference variant(s). 'onnx'=literal upstream ONNX (frozen for the golden gate); 'immich'=open_clip weights through Immich's transform; 'openclip'=open_clip's own transform",
    )
```

After `refs = embed_openclip_refs(...)` (around line 266), merge in the ONNX ref
when requested, and update the ordering line:

```python
    if "onnx" in args.ref:
        refs["onnx"] = embed_onnx(args.model, images, queries)
    refs = {k: refs[k] for k in ("onnx", "immich", "openclip") if k in refs}
```

(Leave `gate_pass` keyed on `"immich"` as-is — the script's verdict semantics
are unchanged; `onnx` is an additional diagnostic column. The golden gate
itself lives in the new test, not this script's verdict.)

- [ ] **Step 3: Lint**

Run: `.venv/bin/python -m ruff check scripts/clip_parity.py`
Expected: no errors (the `S310`/import-order rules: keep imports grouped as shown; `hf_hub_download` over a raw URL avoids `S310`).

- [ ] **Step 4: Commit**

```bash
git add scripts/clip_parity.py
git commit -m "feat(parity): add literal-ONNX reference backend to clip_parity

Adds an onnxruntime backend running the immich-app OpenAI-CLIP ONNX export
through Immich's exact transform/tokenization, selectable via --ref onnx.
This is the reference frozen by the golden-parity gate.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: ONNX reference backend for SigLIP2 (`embedding_parity.py`)

**Files:**
- Modify: `scripts/embedding_parity.py`

SigLIP2 uses different preprocessing (384px squash, 0.5 mean/std) and a SigLIP
tokenizer (not CLIP BPE). The script already constructs `SiglipTextTokenizer`
from a `tokenizer.json` (see `embed_hf`, ~line 303) and already has the SigLIP
image squash in its `immich` variant — reuse both.

- [ ] **Step 1: Read the existing SigLIP transform + tokenizer usage**

Read `scripts/embedding_parity.py` around `embed_hf` (lines ~263–345) and note:
the exact image-pixel function it uses for the `immich` variant, the
`SiglipTextTokenizer` import path, and the `HF_REPO` constant. The ONNX backend
must feed byte-identical inputs.

- [ ] **Step 2: Add the SigLIP2 ONNX backend**

Insert after `embed_hf`. Use the immich-app webli repo; resolve input names
dynamically (SigLIP visual input = pixel_values, textual = input_ids, but read
them from the session to be robust):

```python
ONNX_REPO_SIGLIP2 = "immich-app/ViT-SO400M-16-SigLIP2-384__webli"


def embed_onnx_siglip2(images: list[tuple[str, bytes]], queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Literal upstream SigLIP2 ONNX export (the checkpoint the production index
    was built with) via onnxruntime, fed Immich's exact transform + tokenizer.

    Returns (image_embeds [N,D], text_embeds [M,D]), L2-normalized float32.
    """
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    from src.models.immich_preprocess import siglip_image_pixels  # 384/0.5 defaults
    # Reuse the SAME tokenizer the immich variant uses (see embed_hf):
    from src.models.siglip_tokenizer import SiglipTextTokenizer  # confirm import path in Step 1

    vis_path = hf_hub_download(ONNX_REPO_SIGLIP2, "visual/model.onnx")
    txt_path = hf_hub_download(ONNX_REPO_SIGLIP2, "textual/model.onnx")
    tok_json = hf_hub_download(ONNX_REPO_SIGLIP2, "tokenizer.json")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    vis = ort.InferenceSession(vis_path, so, providers=["CPUExecutionProvider"])
    txt = ort.InferenceSession(txt_path, so, providers=["CPUExecutionProvider"])
    vis_in = vis.get_inputs()[0].name
    txt_in = txt.get_inputs()[0].name
    tokenizer = SiglipTextTokenizer(tok_json)

    def _l2(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return (v / n if n > 0 else v).astype(np.float32)

    img_out = []
    for _, b in images:
        pil = Image.open(io.BytesIO(b)).convert("RGB")
        px = siglip_image_pixels(pil).astype(np.float32)  # SigLIP2 defaults: 384, 0.5
        img_out.append(_l2(vis.run(None, {vis_in: px})[0][0]))
    txt_out = []
    for q in queries:
        ids = np.asarray(tokenizer(q), dtype=np.int64).reshape(1, -1)  # confirm tokenizer() output shape in Step 1
        txt_out.append(_l2(txt.run(None, {txt_in: ids})[0][0]))
    del vis, txt
    gc.collect()
    return np.stack(img_out), np.stack(txt_out)
```

> **Implementation note:** In Step 1 confirm (a) the exact `SiglipTextTokenizer`
> import path and its call signature / return shape, and (b) the textual input
> dtype the ONNX expects (int64 vs int32) via `txt.get_inputs()[0].type`. Adjust
> the two marked lines to match. If `gc`/`io` aren't already imported in this
> script, add them.

- [ ] **Step 3: Wire `onnx` into `--ref`**

In `main()` (around line 388) add `"onnx"` to the `--ref` choices, and after the
HF refs are built (around line 431) merge it in and reorder:

```python
    if "onnx" in args.ref:
        refs["onnx"] = embed_onnx_siglip2(images, queries)
    refs = {k: refs[k] for k in ("onnx", "immich", "transformers", "openclip") if k in refs}
```

- [ ] **Step 4: Lint**

Run: `.venv/bin/python -m ruff check scripts/embedding_parity.py`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/embedding_parity.py
git commit -m "feat(parity): add literal-ONNX SigLIP2 reference backend

Adds an onnxruntime backend running the immich-app SigLIP2 webli ONNX export
through Immich's exact transform/tokenizer, selectable via --ref onnx. This is
the reference frozen by the golden-parity gate for the default smart-search model.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Factor the face upstream-ONNX pass for golden freezing

**Files:**
- Modify: `scripts/face_embedding_parity.py`

The script already has `make_upstream_detector`, `upstream_faces` (SCRFD ONNX),
and `embed` (ArcFace ONNX). Add one helper that returns, per upstream-detected
face on a `Sample`, the frozen-golden tuple the generator and test need.

- [ ] **Step 1: Add `upstream_embeddings()`**

Insert after `embed()` (around line 308):

```python
def upstream_embeddings(samples: list[Sample], det_size: int = 640) -> list[dict]:
    """Run the upstream (SCRFD detect → ArcFace embed) ONNX pipeline on each
    sample. Returns one record per detected face:
        {"img_id": int, "name": str, "label": str|None,
         "bbox": (x1,y1,x2,y2), "embedding": np.ndarray[512] (L2-normalized)}
    This is exactly the golden the face gate compares the MLX/Apple-Vision fork
    pipeline against.
    """
    app = make_upstream_detector(det_size)
    records: list[dict] = []
    for img_id, s in enumerate(samples):
        img_bgr = cv2.imdecode(np.frombuffer(s.data, np.uint8), cv2.IMREAD_COLOR)
        for f in upstream_faces(app, img_bgr):
            emb = embed(s.data, f["kps"])
            n = float(np.linalg.norm(emb))
            records.append({
                "img_id": img_id,
                "name": s.name,
                "label": s.label,
                "bbox": tuple(float(v) for v in f["bbox"]),
                "embedding": (emb / n if n > 0 else emb).astype(np.float32),
            })
    return records
```

> **Implementation note:** confirm `cv2` is already imported in this script
> (`upstream_faces` takes a BGR array, so the caller must decode bytes → BGR).
> If `cv2` isn't imported, add `import cv2`. If a sibling decode helper already
> exists, use it instead.

- [ ] **Step 2: Lint**

Run: `.venv/bin/python -m ruff check scripts/face_embedding_parity.py`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/face_embedding_parity.py
git commit -m "refactor(parity): factor upstream ArcFace embeddings for golden freeze

Adds upstream_embeddings(): runs the SCRFD-detect/ArcFace-embed ONNX pipeline
per face and returns frozen-golden records (bbox, label, img_id, embedding) the
golden generator and face gate consume.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Golden generator — CLIP goldens

**Files:**
- Create: `scripts/gen_parity_golden.py`

Run-once. Loads committed CLIP fixtures + `queries.txt`, calls the two ONNX
backends, writes `.npz` (embeddings) + `.json` (manifest). Face is added in
Task 7. Scripts are loaded by path elsewhere, but this one runs as
`.venv/bin/python scripts/gen_parity_golden.py` from `ml/`, so `import`ing the
sibling scripts needs the scripts dir on `sys.path`.

- [ ] **Step 1: Write the generator (CLIP portion)**

```python
#!/usr/bin/env python3
"""Generate committed golden parity references from the upstream ONNX models.

Run-once, locally (needs HF download + onnxruntime). Output is committed:
  tests/fixtures/golden/{openai_clip,siglip2,face}.npz + .json

Usage:
  .venv/bin/python scripts/gen_parity_golden.py --targets openai_clip siglip2 face
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import onnxruntime as ort

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "scripts"))

FIX = ML_ROOT / "tests" / "fixtures"
GOLDEN = FIX / "golden"

OPENAI_MODEL = "ViT-B-32__openai"


def _load_clip_images() -> list[tuple[str, bytes]]:
    files = sorted((FIX / "clip").glob("*.jpg"))
    if not files:
        raise SystemExit(f"No CLIP fixtures in {FIX / 'clip'}; run fetch_pd_clip_fixtures.py first.")
    return [(f.name, f.read_bytes()) for f in files]


def _load_queries() -> list[str]:
    return [ln.strip() for ln in (FIX / "queries.txt").read_text().splitlines() if ln.strip()]


def _check_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise SystemExit(f"{name}: non-finite values in golden embeddings — refusing to commit.")
    norms = np.linalg.norm(arr, axis=1)
    if (norms < 1e-6).any():
        raise SystemExit(f"{name}: zero-norm embedding in golden — refusing to commit.")


def _write(stem: str, npz: dict, manifest: dict) -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    np.savez(GOLDEN / f"{stem}.npz", **npz)
    (GOLDEN / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[golden] wrote {stem}.npz + {stem}.json")


def gen_openai_clip() -> None:
    from clip_parity import _ONNX_REPO, embed_onnx

    images = _load_clip_images()
    queries = _load_queries()
    img, txt = embed_onnx(OPENAI_MODEL, images, queries)
    _check_finite("openai_clip image", img)
    _check_finite("openai_clip text", txt)
    _write(
        "openai_clip",
        {"image_embeds": img, "text_embeds": txt},
        {
            "model": OPENAI_MODEL,
            "onnx_repo": _ONNX_REPO[OPENAI_MODEL][0],
            "onnx_repo_commit": "FILL_IN_resolved_sha",
            "onnxruntime_version": ort.__version__,
            "dim": int(img.shape[1]),
            "images": [n for n, _ in images],
            "queries": queries,
            "preprocess": "siglip_image_pixels @224 + CLIP mean/std; clean_text(canonicalize=False) + open_clip ViT-B-32 tokenizer",
            "generated": date.today().isoformat(),
        },
    )


def gen_siglip2() -> None:
    from embedding_parity import ONNX_REPO_SIGLIP2, embed_onnx_siglip2

    images = _load_clip_images()
    queries = _load_queries()
    img, txt = embed_onnx_siglip2(images, queries)
    _check_finite("siglip2 image", img)
    _check_finite("siglip2 text", txt)
    _write(
        "siglip2",
        {"image_embeds": img, "text_embeds": txt},
        {
            "model": "ViT-SO400M-16-SigLIP2-384__webli",
            "onnx_repo": ONNX_REPO_SIGLIP2,
            "onnx_repo_commit": "FILL_IN_resolved_sha",
            "onnxruntime_version": ort.__version__,
            "dim": int(img.shape[1]),
            "images": [n for n, _ in images],
            "queries": queries,
            "preprocess": "siglip_image_pixels @384 + 0.5 mean/std; SiglipTextTokenizer",
            "generated": date.today().isoformat(),
        },
    )


# gen_face() added in Task 7.
GENERATORS = {"openai_clip": gen_openai_clip, "siglip2": gen_siglip2}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=list(GENERATORS), choices=list(GENERATORS))
    args = ap.parse_args()
    for t in args.targets:
        print(f"\n=== generating golden: {t} ===")
        GENERATORS[t]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint**

Run: `.venv/bin/python -m ruff check scripts/gen_parity_golden.py`
Expected: no errors.

- [ ] **Step 3: Commit (script only — goldens generated in Task 13)**

```bash
git add scripts/gen_parity_golden.py
git commit -m "feat(parity): add golden-reference generator (CLIP targets)

Run-once generator that freezes upstream-ONNX CLIP embeddings (SigLIP2 +
OpenAI-CLIP) over committed fixtures into tests/fixtures/golden/, with a manifest
and finite/zero-norm guards so a degenerate golden can't be committed silently.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Golden generator — face + commit LFW fixtures

**Files:**
- Modify: `scripts/gen_parity_golden.py`
- Create: `tests/fixtures/faces/<id>/*.jpg` (output)

The face generator downloads the LFW subset, **saves the selected images** into
`tests/fixtures/faces/<identity>/` (so the test is hermetic), runs the upstream
pipeline, computes the golden top-1 accuracy, and freezes everything.

- [ ] **Step 1: Add `gen_face()` and register it**

Insert before `GENERATORS`:

```python
FACE_NUM_IDS = 3
FACE_MIN_PER_ID = 2
FACE_MAX_PER_ID = 2


def _save_face_fixtures(samples) -> None:
    """Persist the LFW subset under tests/fixtures/faces/<label>/ (committed)."""
    root = FIX / "faces"
    for s in samples:
        # s.name is "<path>" or "<label>.jpg"; group by identity label.
        ident = (s.label or "unknown").replace(" ", "_")
        d = root / ident
        d.mkdir(parents=True, exist_ok=True)
        fname = Path(s.name).name
        (d / fname).write_bytes(s.data)


def gen_face() -> None:
    from face_embedding_parity import load_lfw, top1_accuracy, upstream_embeddings

    samples = load_lfw(FACE_NUM_IDS, FACE_MIN_PER_ID, FACE_MAX_PER_ID)
    _save_face_fixtures(samples)
    records = upstream_embeddings(samples)
    if not records:
        raise SystemExit("face: upstream pipeline detected zero faces — cannot freeze golden.")
    emb = np.stack([r["embedding"] for r in records])
    _check_finite("face", emb)
    labels = [r["label"] for r in records]
    img_ids = [r["img_id"] for r in records]
    bboxes = np.array([r["bbox"] for r in records], dtype=np.float32)
    # Golden top-1 retrieval accuracy of the upstream embeddings against itself
    # (same-image excluded) — the test asserts MLX is within 0.02 of this.
    golden_top1 = top1_accuracy(emb, labels, img_ids, emb, labels, img_ids)
    _write(
        "face",
        {
            "embeddings": emb,
            "bboxes": bboxes,
            "labels": np.array(labels),
            "img_ids": np.array(img_ids),
            "golden_top1": np.array(golden_top1, dtype=np.float64),
        },
        {
            "model": "buffalo_l (SCRFD det_10g + ArcFace w600k_r50)",
            "onnxruntime_version": ort.__version__,
            "dim": int(emb.shape[1]),
            "num_faces": len(records),
            "identities": sorted({r["label"] for r in records}),
            "golden_top1_accuracy": float(golden_top1),
            "generated": date.today().isoformat(),
        },
    )
```

Then update the registry line:

```python
GENERATORS = {"openai_clip": gen_openai_clip, "siglip2": gen_siglip2, "face": gen_face}
```

> **Implementation note:** confirm `top1_accuracy`'s exact signature in
> `face_embedding_parity.py` (it takes query and gallery args). Passing the same
> arrays for query and gallery with same-image exclusion is what the harness test
> `test_top1_excludes_same_image` already exercises.

- [ ] **Step 2: Lint**

Run: `.venv/bin/python -m ruff check scripts/gen_parity_golden.py`
Expected: no errors.

- [ ] **Step 3: Commit (script only)**

```bash
git add scripts/gen_parity_golden.py
git commit -m "feat(parity): add face golden generator + LFW fixture persistence

gen_face() downloads a small LFW subset, persists it under tests/fixtures/faces/
(so the gate is hermetic), runs the upstream SCRFD/ArcFace ONNX pipeline, and
freezes embeddings + bboxes + labels + golden top-1 accuracy.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `parity_gate()` helper + hermetic unit test

**Files:**
- Create: `tests/_parity_gate.py`
- Create: `tests/test_parity_gate.py`

Auto-detect dependency/golden availability; `ML_RUN_PARITY=1` turns "unavailable"
into a hard failure rather than a skip.

- [ ] **Step 1: Write the failing unit test**

```python
"""Hermetic tests for the parity-gate skip/hard-fail decision logic."""
import importlib

import pytest

from tests import _parity_gate as g


def test_available_returns_run(monkeypatch):
    monkeypatch.setattr(g, "_deps_available", lambda: (True, ""))
    monkeypatch.setattr(g, "_golden_present", lambda stem: True)
    assert g.gate_reason("openai_clip") is None


def test_missing_deps_skips_by_default(monkeypatch):
    monkeypatch.delenv("ML_RUN_PARITY", raising=False)
    monkeypatch.setattr(g, "_deps_available", lambda: (False, "no mlx"))
    assert g.gate_reason("openai_clip") == "no mlx"


def test_missing_golden_skips_by_default(monkeypatch):
    monkeypatch.delenv("ML_RUN_PARITY", raising=False)
    monkeypatch.setattr(g, "_deps_available", lambda: (True, ""))
    monkeypatch.setattr(g, "_golden_present", lambda stem: False)
    reason = g.gate_reason("face")
    assert reason is not None and "golden" in reason.lower()


def test_forced_missing_is_hard_fail(monkeypatch):
    monkeypatch.setenv("ML_RUN_PARITY", "1")
    monkeypatch.setattr(g, "_deps_available", lambda: (False, "no onnxruntime"))
    with pytest.raises(g.ParityUnavailable, match="no onnxruntime"):
        g.gate_reason("openai_clip")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_parity_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: tests._parity_gate`.

- [ ] **Step 3: Write `tests/_parity_gate.py`**

```python
"""Decide whether weights-gated parity tests run, skip, or hard-fail.

Default: auto-detect (skip if deps or golden artifacts are unavailable).
ML_RUN_PARITY=1: a missing prerequisite is a HARD FAILURE, so CI can guarantee
the gate actually executed instead of silently skipping.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
_REQUIRED = ("mlx", "onnxruntime", "numpy", "PIL")


class ParityUnavailable(RuntimeError):
    """Raised when ML_RUN_PARITY=1 but prerequisites are missing."""


def _deps_available() -> tuple[bool, str]:
    for mod in _REQUIRED:
        if importlib.util.find_spec(mod) is None:
            return False, f"missing dependency: {mod}"
    return True, ""


def _golden_present(stem: str) -> bool:
    return (GOLDEN_DIR / f"{stem}.npz").is_file()


def gate_reason(stem: str) -> str | None:
    """Return None to RUN, or a skip-reason string to SKIP.

    With ML_RUN_PARITY=1, an unavailable prerequisite raises ParityUnavailable
    (a hard failure) instead of returning a skip reason.
    """
    forced = os.getenv("ML_RUN_PARITY") == "1"
    ok, reason = _deps_available()
    if not ok:
        if forced:
            raise ParityUnavailable(reason)
        return reason
    if not _golden_present(stem):
        reason = f"golden artifact missing: {stem}.npz (run scripts/gen_parity_golden.py)"
        if forced:
            raise ParityUnavailable(reason)
        return reason
    return None
```

Also ensure `tests/__init__.py` exists (it does — confirmed in the tests dir
listing). If not, create an empty one so `from tests import _parity_gate` works.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_parity_gate.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/_parity_gate.py tests/test_parity_gate.py
git commit -m "test(parity): add gate helper (auto-detect skip + ML_RUN_PARITY hard-fail)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: CLIP golden-parity gated test

**Files:**
- Create: `tests/test_clip_golden_parity.py`

Parametrized over the two CLIP models. Loads the MLX production backend, embeds
committed fixtures + queries, asserts per-item min cosine ≥ 0.99 vs golden.

- [ ] **Step 1: Write the test**

```python
"""Weights-gated CLIP parity: MLX production backend vs committed ONNX golden.

Skips unless prerequisites + golden artifacts are present (ML_RUN_PARITY=1 makes
that a hard failure). The conftest forces STUB_MODE=true for hermetic unit tests;
these tests load REAL models, so they run only on the opt-in/auto-detected path.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests._parity_gate import gate_reason

FIX = Path(__file__).resolve().parent / "fixtures"
GOLDEN = FIX / "golden"
THRESHOLD = 0.99

# (golden stem, Immich model name routed to the MLX backend)
CASES = [
    ("openai_clip", "ViT-B-32__openai"),
    ("siglip2", "ViT-SO400M-16-SigLIP2-384__webli"),
]


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)


def _load_fixtures():
    images = [(f.name, f.read_bytes()) for f in sorted((FIX / "clip").glob("*.jpg"))]
    queries = [ln.strip() for ln in (FIX / "queries.txt").read_text().splitlines() if ln.strip()]
    return images, queries


@pytest.mark.parametrize("stem,model_name", CASES, ids=[c[0] for c in CASES])
def test_clip_mlx_matches_onnx_golden(stem, model_name):
    reason = gate_reason(stem)
    if reason is not None:
        pytest.skip(reason)

    # Real models required; opt out of the conftest's forced STUB_MODE.
    os.environ["STUB_MODE"] = "false"

    gold = np.load(GOLDEN / f"{stem}.npz")
    g_img, g_txt = gold["image_embeds"], gold["text_embeds"]
    images, queries = _load_fixtures()
    assert len(images) == g_img.shape[0], "fixture/golden image count drift — regenerate golden"
    assert len(queries) == g_txt.shape[0], "fixture/golden query count drift — regenerate golden"

    from src.models.clip import get_clip_model

    forced = os.getenv("ML_RUN_PARITY") == "1"
    try:
        model = get_clip_model(model_name)
        mlx_img = np.stack([model.encode_image(b) for _, b in images])
        mlx_txt = np.stack([model.encode_text(q) for q in queries])
        model.unload()
    except Exception as e:  # noqa: BLE001 — translate to skip unless forced
        if forced:
            raise
        pytest.skip(f"MLX backend for {model_name} unavailable: {e}")

    img_cos = _cosine_rows(mlx_img, g_img)
    txt_cos = _cosine_rows(mlx_txt, g_txt)
    img_fail = [(images[i][0], float(img_cos[i])) for i in range(len(images)) if img_cos[i] < THRESHOLD]
    txt_fail = [(queries[j], float(txt_cos[j])) for j in range(len(queries)) if txt_cos[j] < THRESHOLD]
    assert not img_fail, f"{model_name}: image cosine below {THRESHOLD}: {img_fail}"
    assert not txt_fail, f"{model_name}: text cosine below {THRESHOLD}: {txt_fail}"
```

- [ ] **Step 2: Run it (auto-detect skip path)**

Run: `.venv/bin/python -m pytest tests/test_clip_golden_parity.py -v`
Expected: 2 SKIPPED (golden artifacts don't exist yet — they're generated in Task 13). This confirms the gate wiring works without weights.

- [ ] **Step 3: Commit**

```bash
git add tests/test_clip_golden_parity.py
git commit -m "test(parity): add weights-gated CLIP golden-reference gate

Fails if MLX CLIP embeddings (SigLIP2 + OpenAI-CLIP) drift below cosine 0.99
versus the committed upstream-ONNX golden references.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Face golden-parity gated test

**Files:**
- Create: `tests/test_face_golden_parity.py`

Runs the MLX/Apple-Vision fork pipeline on committed LFW fixtures, greedy-matches
to golden faces by bbox IoU, asserts median drift cosine ≥ 0.90 and top-1 drop ≤
0.02. Apple Vision is macOS-only, so the gate also skips off-macOS.

- [ ] **Step 1: Write the test**

```python
"""Weights-gated face parity: MLX/Apple-Vision fork vs committed upstream golden.

Asserts alignment-drift median cosine >= 0.90 and top-1 retrieval drop <= 0.02.
Apple Vision (the fork detector) is macOS-only; skips elsewhere.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._parity_gate import ParityUnavailable, gate_reason

FIX = Path(__file__).resolve().parent / "fixtures"
GOLDEN = FIX / "golden"
MEDIAN_COS_MIN = 0.90
TOP1_DROP_MAX = 0.02


def _require_macos():
    if sys.platform != "darwin":
        if os.getenv("ML_RUN_PARITY") == "1":
            raise ParityUnavailable("face parity needs Apple Vision (macOS only)")
        pytest.skip("face parity needs Apple Vision (macOS only)")


def test_face_mlx_matches_onnx_golden():
    reason = gate_reason("face")
    if reason is not None:
        pytest.skip(reason)
    _require_macos()
    os.environ["STUB_MODE"] = "false"

    # Load script helpers by path (scripts/ is not a package).
    spec = importlib.util.spec_from_file_location(
        "face_embedding_parity",
        Path(__file__).resolve().parents[1] / "scripts" / "face_embedding_parity.py",
    )
    fep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fep)

    gold = np.load(GOLDEN / "face.npz", allow_pickle=False)
    g_emb = gold["embeddings"]
    g_bbox = gold["bboxes"]
    g_labels = list(gold["labels"])
    g_img_ids = list(int(x) for x in gold["img_ids"])
    golden_top1 = float(gold["golden_top1"])

    samples = fep.load_image_dir(FIX / "faces")
    forced = os.getenv("ML_RUN_PARITY") == "1"

    # Fork pipeline: Apple Vision detect → ArcFace embed, per face.
    mlx_emb, mlx_bbox, mlx_labels, mlx_img_ids = [], [], [], []
    try:
        for img_id, s in enumerate(samples):
            for f in fep.fork_faces(s.data, nose_strategy="tip"):
                e = fep.embed(s.data, f["kps"])
                n = float(np.linalg.norm(e))
                mlx_emb.append((e / n if n > 0 else e).astype(np.float32))
                mlx_bbox.append(f["bbox"])
                mlx_labels.append(s.label)
                mlx_img_ids.append(img_id)
    except Exception as e:  # noqa: BLE001
        if forced:
            raise
        pytest.skip(f"MLX face pipeline unavailable: {e}")

    assert mlx_emb, "fork pipeline detected zero faces on committed fixtures"
    mlx_emb_arr = np.stack(mlx_emb)

    # Greedy-match fork faces to golden faces by bbox IoU (same image only).
    drift = []
    for gi in range(len(g_labels)):
        best_j, best_iou = -1, 0.0
        for j in range(len(mlx_bbox)):
            if mlx_img_ids[j] != g_img_ids[gi]:
                continue
            iou = fep.iou(tuple(g_bbox[gi]), tuple(mlx_bbox[j]))
            if iou > best_iou:
                best_iou, best_j = iou, j
        assert best_j >= 0 and best_iou >= 0.3, (
            f"golden face {gi} ({g_labels[gi]}, img {g_img_ids[gi]}) has no fork match "
            f"(best IoU {best_iou:.2f}) — detection-set regression"
        )
        drift.append(float(np.dot(g_emb[gi], mlx_emb_arr[best_j])))

    median_cos = float(np.median(drift))
    assert median_cos >= MEDIAN_COS_MIN, (
        f"median alignment-drift cosine {median_cos:.4f} < {MEDIAN_COS_MIN}; per-face={sorted(drift)}"
    )

    mlx_top1 = fep.top1_accuracy(mlx_emb_arr, mlx_labels, mlx_img_ids, mlx_emb_arr, mlx_labels, mlx_img_ids)
    drop = golden_top1 - float(mlx_top1)
    assert drop <= TOP1_DROP_MAX, (
        f"top-1 retrieval dropped {drop:.4f} (golden {golden_top1:.4f} -> mlx {mlx_top1:.4f}); max {TOP1_DROP_MAX}"
    )
```

> **Implementation note:** confirm `fep.iou`, `fep.fork_faces`, `fep.embed`,
> `fep.top1_accuracy`, and `fep.load_image_dir` signatures match Task 5's reading.
> The greedy match here is inline (golden vs fork) rather than `greedy_match`
> (which matches two live detector lists); both use the same IoU helper and 0.3
> threshold.

- [ ] **Step 2: Run it (auto-detect skip path)**

Run: `.venv/bin/python -m pytest tests/test_face_golden_parity.py -v`
Expected: 1 SKIPPED (golden missing until Task 13).

- [ ] **Step 3: Commit**

```bash
git add tests/test_face_golden_parity.py
git commit -m "test(parity): add weights-gated face golden-reference gate

Runs the MLX/Apple-Vision fork pipeline against the committed upstream-ONNX
golden: median alignment-drift cosine >= 0.90 and top-1 retrieval drop <= 0.02.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Rename harness-only test files

**Files:**
- Rename: `tests/test_embedding_parity.py` → `tests/test_embedding_parity_harness.py`
- Rename: `tests/test_face_embedding_parity.py` → `tests/test_face_embedding_parity_harness.py`

- [ ] **Step 1: Rename with git (preserves history)**

```bash
git mv tests/test_embedding_parity.py tests/test_embedding_parity_harness.py
git mv tests/test_face_embedding_parity.py tests/test_face_embedding_parity_harness.py
```

- [ ] **Step 2: Update each file's module docstring**

In both files, prepend a one-line note to the existing docstring clarifying scope,
e.g. for `test_embedding_parity_harness.py`:

```python
"""Harness math for scripts/embedding_parity.py (NOT model parity).

Model-output parity is gated by tests/test_clip_golden_parity.py against
committed ONNX golden references; this file covers only the helper math
(download retry, cosine, stats, retrieval_agreement).
"""
```

And analogously for `test_face_embedding_parity_harness.py` (point at
`test_face_golden_parity.py`). Keep the rest of each docstring/body unchanged.

- [ ] **Step 3: Run the renamed tests to confirm nothing broke**

Run: `.venv/bin/python -m pytest tests/test_embedding_parity_harness.py tests/test_face_embedding_parity_harness.py -v`
Expected: all PASS (same tests as before, new filenames).

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test(parity): rename harness-only parity tests to *_harness

These cover only helper math (cosine/IoU/retrieval), not model output. The new
*_golden_parity.py files are the real weights-gated model-parity gate; renaming
stops the old names implying parity coverage.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Hermetic suite check

**Files:** none (verification)

- [ ] **Step 1: Run the full ml/ suite (default hermetic mode)**

Run: `.venv/bin/python -m pytest`
Expected: all prior tests PASS; the 3 new gated tests (`test_clip_golden_parity`
×2, `test_face_golden_parity` ×1) SKIP with golden-missing/macOS reasons;
`test_parity_gate` PASSes. No failures, no errors. (Don't pipe through `tail` —
read the real summary line.)

- [ ] **Step 2: Confirm exit code**

Run: `.venv/bin/python -m pytest; echo "exit=$?"`
Expected: `exit=0`.

---

## Task 13: Generate goldens locally + verify the gate passes (local, weights)

**Files:**
- Create: `tests/fixtures/golden/*.npz`, `tests/fixtures/golden/*.json`
- Create: `tests/fixtures/faces/<id>/*.jpg`
- Modify: `tests/fixtures/README.md` (face provenance, golden manifests note)

This task needs real model downloads (large, esp. SigLIP2 ~3.5 GB) and macOS for
the face leg. Per repo policy, verify locally before claiming it works.

- [ ] **Step 1: Generate all goldens**

Run: `.venv/bin/python scripts/gen_parity_golden.py --targets openai_clip siglip2 face`
Expected: writes `tests/fixtures/golden/{openai_clip,siglip2,face}.npz` + `.json`
and populates `tests/fixtures/faces/<id>/`. No "non-finite"/"zero-norm"/"zero
faces" errors. If the SigLIP2 textual dtype/tokenizer note from Task 4 was wrong,
fix it now and rerun.

- [ ] **Step 2: Fill in the resolved ONNX commit SHAs**

For each CLIP manifest, replace `"FILL_IN_resolved_sha"` with the resolved repo
revision. Get it via:
```bash
.venv/bin/python - <<'PY'
from huggingface_hub import HfApi
for r in ("immich-app/ViT-B-32__openai", "immich-app/ViT-SO400M-16-SigLIP2-384__webli"):
    print(r, HfApi().repo_info(r).sha)
PY
```

- [ ] **Step 3: Run the gated tests with hard-fail enforcement**

Run: `ML_RUN_PARITY=1 .venv/bin/python -m pytest tests/test_clip_golden_parity.py tests/test_face_golden_parity.py -v`
Expected: all 3 PASS (CLIP ×2 cosine ≥ 0.99; face median ≥ 0.90 and top-1 drop ≤
0.02). If a CLIP case fails near-zero cosine, that's a real weights/activation
mismatch in the MLX backend — stop and investigate (do not relax the threshold).

- [ ] **Step 4: Confirm committed size is small**

Run: `du -sh tests/fixtures` and `git status --short tests/fixtures`
Expected: total well under ~1 MB. If a CLIP fixture or LFW image is large, the
fetch/gen resize step needs tightening.

- [ ] **Step 5: Finalize README provenance**

Add the LFW identities + source to the `faces/` section and note the manifests
under `golden/` in `tests/fixtures/README.md`.

- [ ] **Step 6: Commit goldens + fixtures**

```bash
git add tests/fixtures/golden tests/fixtures/faces tests/fixtures/README.md
git commit -m "test(parity): commit ONNX golden references + LFW face fixtures

Frozen upstream-ONNX embeddings (SigLIP2, OpenAI-CLIP, ArcFace) + a small LFW
subset. Verified locally: CLIP cosine >= 0.99, face median cosine >= 0.90,
top-1 drop <= 0.02 under ML_RUN_PARITY=1.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Docs + parent submodule pointer bump

**Files:**
- Modify: `README.md` (ml submodule)
- Modify (parent repo): submodule pointer

- [ ] **Step 1: Document the gate in the ml README**

Add a short subsection near the existing parity sections (after "CLIP parity"
/ "Parity-or-fail") describing the automated gate: what it checks (SigLIP2 +
OpenAI-CLIP cosine ≥ 0.99; face median ≥ 0.90, top-1 drop ≤ 0.02), where the
fixtures/goldens live (`tests/fixtures/`), how to regenerate
(`scripts/gen_parity_golden.py`), and how to run it
(`ML_RUN_PARITY=1 .venv/bin/python -m pytest tests/test_*_golden_parity.py`).
Note that the default `pytest` run skips it (hermetic), and CI sets
`ML_RUN_PARITY=1`.

- [ ] **Step 2: Commit the README**

```bash
git add README.md
git commit -m "docs(parity): document the automated golden-reference parity gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Bump the parent submodule pointer**

From the **parent worktree root** (not `ml/`):
```bash
cd /Users/john/Code/immich-apple-silicon/.claude/worktrees/cheeky-jumping-hinton
git add ml
git commit -m "chore: bump ml submodule pointer — committed CLIP/face parity golden gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

> Push/sync is handled per the worktree flow in CLAUDE.md at session close (push
> `ml` first, then the parent), not mid-task.

---

## Self-review notes

- **Spec coverage:** committed fixtures (Tasks 1,2,7,13) ✓; reference embeddings
  from literal ONNX (Tasks 3,4,5,6,7) ✓; weights-gated CLIP ≥0.99 (Task 9) ✓;
  face median ≥0.90 + top-1 drop ≤0.02 (Task 10) ✓; auto-detect + ML_RUN_PARITY
  hard-fail (Task 8) ✓; harness rename (Task 11) ✓; manifests with repo+SHA+ort
  version (Tasks 6,7,13) ✓; Approach C — ONNX backend in scripts + thin
  generator (Tasks 3–7) ✓; <1 MB committed (Task 13 step 4) ✓.
- **Open implementation confirmations (flagged inline, not blockers):** SigLIP2
  `SiglipTextTokenizer` import path + textual ONNX dtype (Task 4 Step 1); `cv2`
  import in the face script (Task 5); `top1_accuracy` signature (Tasks 7,10);
  `immich-app` ONNX repo layout `visual/textual/model.onnx` (matches the proven
  `clip_benchmark.OnnxBackend`, Task 3); MLX loaders honor `STUB_MODE=false`
  override (Tasks 9,10 — if a loader ignores the env, set it before importing).
- **Type consistency:** golden npz keys are consistent between generator and
  tests — CLIP: `image_embeds`/`text_embeds`; face: `embeddings`/`bboxes`/
  `labels`/`img_ids`/`golden_top1`. `gate_reason(stem)` stem strings
  (`openai_clip`,`siglip2`,`face`) match golden filenames.
