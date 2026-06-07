# CLIP & face parity golden-reference gate — design

**Date:** 2026-06-07
**Scope:** `ml/` submodule (immich-ml-metal)

## Problem

`ml/tests/test_embedding_parity.py` and `test_face_embedding_parity.py` test only
harness math (cosine / IoU / greedy-match / retrieval-agreement) — **not** model
output. No automated test loads MLX weights and compares them to the upstream
ONNX models Immich actually ships, and there are zero committed golden
embeddings. Real parity is validated only by manual scripts
(`scripts/{embedding,clip,face_embedding}_parity.py`) that need a human to pass
`--images`, so a parity regression (wrong weights, a bad activation, a
dependency bump that shifts embeddings) is invisible to automation.

This design adds an **automated, committed-golden parity gate**: weights-gated
tests that fail if MLX embeddings drift below threshold versus reference
embeddings frozen from the literal upstream ONNX models.

## Goals

1. Commit a small, license-clean fixture image set plus reference embeddings
   generated from the **upstream ONNX models Immich actually ships**.
2. Add weights-gated tests asserting:
   - CLIP image **and** text cosine ≥ `0.99` (per-item min), for both the default
     deployed **SigLIP2** model and the vendored **OpenAI-CLIP** path.
   - Face alignment-drift: median cosine ≥ `0.90`, and top-1 retrieval accuracy
     drop ≤ `0.02` (MLX/Apple-Vision fork vs upstream SCRFD/ArcFace golden).
3. Rename the harness-only test files to `*_harness.py` so they stop implying
   parity coverage.

## Non-goals / out of scope

- Enabling `ml/` in CI (tracked separately as the companion CI task). This design
  makes the gate *exist and pass locally*; wiring it into a CI job is follow-up.
- LAION CLIP ports (no parity-faithful MLX backend yet — they raise by design).
- Quantization parity (already covered by `scripts/quantization_eval.py`).

## Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Models gated | SigLIP2 `ViT-SO400M-16-SigLIP2-384__webli` + OpenAI-CLIP `ViT-B-32__openai` + face ArcFace (`buffalo_l`) |
| Golden source | **Literal ONNX** via onnxruntime on the `immich-app` HF repos, with Immich's exact preprocessing/tokenization |
| Fixtures | Face: small **LFW** subset (~3 identities × 2 imgs). CLIP: ~8 **public-domain** photos (Wikimedia PD / NASA) |
| Structure | **Approach C (hybrid):** add an ONNX reference backend to the diagnostic scripts; a thin run-once generator freezes goldens; new gated tests consume them |
| Gating | **Auto-detect by default** (skip if deps/weights unavailable); `ML_RUN_PARITY=1` forces a **hard fail** instead of skip, so CI can guarantee the gate ran |

## Architecture (Approach C)

### 1. Fixtures — `ml/tests/fixtures/`

```
tests/fixtures/
  clip/                 ~8 public-domain photos, resized ≤512px (diverse content)
  faces/<identity>/*.jpg  LFW subset, ~3 identities × ≥2 images
  queries.txt           fixed CLIP text-query list (one per line)
  golden/
    siglip2.npz         image_embeds[N,D], text_embeds[M,D] (L2-normalized)
    siglip2.json        manifest (see below)
    openai_clip.npz
    openai_clip.json
    face.npz            upstream-aligned ArcFace embeds[K,512], bboxes[K,4],
                        labels[K], img_ids[K]
    face.json
  README.md             per-file provenance + license + source URL
```

Total committed size target < ~1 MB (embeddings are tiny; images resized small).

**Manifest (`*.json`)** records, for reproducibility: model name, `immich-app`
ONNX repo + resolved commit SHA, onnxruntime version, embedding dim, fixture
filenames, query list, preprocessing variant, generation date. The date and
SHA are recorded by the human running the generator (not derived at test time).

### 2. ONNX reference backends (added to existing diagnostic scripts)

- **`scripts/clip_parity.py`** — add `embed_onnx(model_name, images, queries)`:
  loads the `immich-app/<model>` visual + textual ONNX via onnxruntime, feeds
  **Immich's exact transform** (`siglip_image_pixels` with CLIP constants +
  resize-shortest-224/center-crop) and the **same CLIP tokenization** the MLX
  backend uses. Returns L2-normalized `(image_embeds, text_embeds)`. Wired as a
  new `--ref onnx` variant alongside `immich`/`openclip`.
- **`scripts/embedding_parity.py`** — add `embed_onnx_siglip2(images, queries)`:
  same idea for `immich-app/ViT-SO400M-16-SigLIP2-384__webli` (SigLIP squash
  transform + SigLIP tokenizer). Wired as `--ref onnx`.
- **`scripts/face_embedding_parity.py`** — already runs the literal upstream ONNX
  (SCRFD `det_10g.onnx` + ArcFace `w600k_r50.onnx`). Factor its upstream pass so
  the generator can call it and freeze `(embedding, bbox, label, img_id)` per
  upstream-detected face.

Reusing `src.models.immich_preprocess` + the existing tokenizers guarantees the
gate isolates **weights/activation drift**, not preprocessing differences.

Benefit of C: the manual diagnostic scripts also gain a literal-ONNX diff (not
just the open_clip/transformers proxies they use today).

### 3. Generator — `scripts/gen_parity_golden.py` (run-once, offline)

Thin wrapper. For each target it calls the script's ONNX backend on the
committed fixtures and writes the `.npz` + `.json` manifest. Run by a human
locally (needs HF download + onnxruntime); output is committed. Records the
resolved ONNX repo commit SHA + onnxruntime version into the manifest.

### 4. Weights-gated tests (new files)

- **`tests/test_clip_golden_parity.py`** — parametrized over `{siglip2,
  openai_clip}`: load the MLX backend (`get_clip_model`), embed the committed
  fixture images + `queries.txt`, compute per-item cosine vs golden, assert
  `min(image_cos) ≥ 0.99` and `min(text_cos) ≥ 0.99`.
- **`tests/test_face_golden_parity.py`** — run the MLX/Apple-Vision fork pipeline
  on the LFW fixtures, greedy-match detected faces to golden faces by bbox IoU,
  assert `median(drift_cos) ≥ 0.90`; compute top-1 retrieval accuracy on MLX
  embeddings and compare to the golden's stored top-1 accuracy, assert the drop
  ≤ `0.02`.

**Gating helper** (shared, e.g. `tests/_parity_gate.py`):

```
def parity_gate():
    """Return None to run, or a skip-reason. Honors ML_RUN_PARITY.

    - Attempts to import deps / confirm weights are loadable.
    - If unavailable and ML_RUN_PARITY=1 -> raise (hard fail: CI guarantees run).
    - If unavailable and ML_RUN_PARITY unset -> return skip reason.
    """
```

Tests call it at module/fixture scope. This keeps the default `pytest` run
hermetic under the conftest's forced `STUB_MODE=true`, while `ML_RUN_PARITY=1`
guarantees the gate actually executes (no silent skip) in CI.

Note: parity tests must run the **real** backends; they explicitly opt out of the
conftest `STUB_MODE` for their scope (confirm `get_clip_model` / face_embed honor
this during implementation).

### 5. Rename harness files

- `tests/test_embedding_parity.py` → `tests/test_embedding_parity_harness.py`
- `tests/test_face_embedding_parity.py` → `tests/test_face_embedding_parity_harness.py`

Shared helper imports they rely on stay importable (they load the scripts by
path). `test_face_align_parity.py` is left as-is (the bead names only the two).

## Data flow

```
[committed fixtures] --(Immich preprocess)--> ONNX backend --(run-once)--> golden .npz  [committed]
                                                                              |
[committed fixtures] --(Immich preprocess)--> MLX backend  --(test time)--> embeds --> cosine/IoU vs golden --> assert thresholds
```

## Error handling

- Generator: fail loudly if a fixture can't be read, an ONNX repo/op is missing,
  or an embedding is non-finite/zero-norm (a degenerate golden must never be
  committed silently).
- Tests: `parity_gate()` distinguishes skip (deps absent, default) from hard
  fail (`ML_RUN_PARITY=1` + deps absent). A below-threshold cosine fails with a
  per-item breakdown (which image/query drifted) for fast triage.
- Face greedy-match: if a golden face has no MLX match above the IoU threshold,
  that is a detection-set regression — surface it explicitly rather than
  silently dropping it from the median.

## Testing the gate itself

- The renamed `*_harness.py` files keep covering the helper math.
- A tiny unit test for `parity_gate()` (env-var → skip vs hard-fail logic) runs
  hermetically without weights.
- Manual: run `gen_parity_golden.py`, then `ML_RUN_PARITY=1 pytest
  tests/test_*_golden_parity.py` locally and confirm green before committing.

## Risks / open items

- **SigLIP2 ONNX download is large** (~3.5 GB). Acceptable for a one-time local
  generation; not run in the default test path.
- **LFW licensing**: research-use dataset; commit a minimal subset and document
  provenance. If this is later deemed unacceptable, fall back to insightface's
  bundled samples (embedding-parity only, no retrieval metric).
- **`immich-app` ONNX repo layout** (visual/textual subdirs, op names) must be
  confirmed at implementation time; pin the resolved commit SHA in the manifest.
- **STUB_MODE override**: confirm the MLX loaders run real models when the parity
  tests opt out of STUB_MODE.

## Acceptance criteria

- A weights-gated test fails if MLX CLIP embeddings drift below cosine `0.99`
  versus committed ONNX golden references (both SigLIP2 and OpenAI-CLIP).
- A weights-gated face test enforces median cos ≥ `0.90` and top-1 drop ≤ `0.02`.
- `ML_RUN_PARITY=1` makes a missing-weights situation a hard failure, not a skip.
- The harness-only files are renamed to `*_harness.py`.
- Committed fixtures + goldens are < ~1 MB with documented provenance/licenses.

## Follow-ups (separate beads)

- Wire these gated tests into `ml/` CI (the companion CI task) with
  `ML_RUN_PARITY=1` on a macOS arm64 runner.
- Optionally extend to ViT-B-16 / L-14 OpenAI ports once needed.
