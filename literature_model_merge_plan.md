# Literature Model Merge Plan — Auto-Annotator + XGBoost Morphokinetic Pipeline → Unified `morpheus_v1` Repo

**Author:** AI Software Architect analysis
**Date:** 2026-06-16
**Subject codebase:** `/home/phd2/Scrivania/CorsoRepo/blasto_annotator_and_prediction` (literature-based blastocyst prediction: NASNetLarge frame auto-annotator → hand-crafted morphokinetic features → XGBoost)
**Target:** Merge into the unified "Paper Release" PyTorch repo (`morpheus_v1/`) that uses `.env`-driven paths, a central patient-level stratified CSV split, pure `nn.Module` classes, and a shared evaluation layer — alongside the time-series models (ROCKET / LSTMFCN / ConvTran, from `cellPIV/`) and the vision models (ViT / ViViT / ResNet, from `blasto_prediction/`).

> **Scope note:** this is an **architectural analysis and merge plan only**. No refactored code is written here. The companion documents are `cellPIV/...` (time series) and [`blasto_prediction/vision_codebase_merge_plan.md`](../blasto_prediction/vision_codebase_merge_plan.md) (vision). This plan deliberately mirrors that vision plan's structure so the three feed one merge.

---

## 1. Executive Summary

### 1.1 What this codebase is
A **research / experiment-grade, notebook-only** reimplementation of a **published "auto-annotation + gradient-boosting" morphokinetic pipeline** for blastocyst-formation prediction. It is **not a neural network** in the end-to-end sense: it is a **three-stage hybrid** in which only the first stage is a deep model.

| Stage | Component | Technology | Role |
|------|-----------|-----------|------|
| **1. Annotate** | `NASNet_embryo_17class` | **TensorFlow/Keras** `NASNetLarge` SavedModel | Per-frame classification of embryo developmental stage into **17 classes** (`1cell`, `1cell_PN`, `2cell_G1`…`8cell_G3`) |
| **2. Featurize** | `convert_to_wide_format` + `process_df_data` | pandas / numpy | Pivot per-frame annotations onto a 15-min time grid (24h→64h/72h) and derive **cell-stage ratio features** + maternal age |
| **3. Predict** | `xgb_<target>.model` | **XGBoost** Booster | Map the engineered feature vector to a blastocyst-outcome probability for one of three targets |

The literature provenance is implicit (Google-Drive-hosted weights, `/content/...` Colab paths in `sample_img_list.csv`), and **no model card / citation / training manifest is committed** — a provenance gap to close on merge.

### 1.2 Two parallel branches
The repo ships the same pipeline twice:

| Folder | Variant | Distinguishing traits |
|--------|---------|----------------------|
| [_01_annotator_and_xgboost/](_01_annotator_and_xgboost/) | **Baseline** | Multi-target loop over all three `df_config` targets; window `24→64h`; **performs grade-unification** of the time-grid labels; ships a sample-image demo (`sample_img_list.csv`, `/content/...` paths); contains the 1 GB `Blastocyst_Prediction.zip` + a stray `.part` |
| [_02_annotator_and_xgboost_custom/](_02_annotator_and_xgboost_custom/) | **Custom** | Single hardcoded target (`blastocyst_prediction`); window `24→72h`; strict whitelist filter on a metadata CSV; **adds bootstrap CIs** (200 iters → AUC/F1/Acc/BalAcc/MCC/Brier) and `model_comparison_line.csv` export; **omits grade-unification** (see §3.4 — a correctness divergence) |

### 1.3 State and cleanliness (honest)
The pipeline **runs and produces the paper-style metrics**, but its engineering posture is **below** even the vision repo's bar and far below the clean repo's:

- **Zero `.py` modules.** All logic lives in two `.ipynb` files. There is no package, no importable surface, no tests.
- **Hardcoded paths + a hardcoded Google-Drive `FILE_ID`** for a ~1 GB weights zip (`gdown`). No `.env`, no `os.getenv`.
- **The "model" is not isolated** — preprocessing, TF inference, pandas feature engineering, XGBoost prediction, metrics, and bootstrap are interleaved across notebook cells operating on **mutable global DataFrames** (`df_files`, `df_dataset`).
- **Mixed stack collision with the target:** the annotator is **TensorFlow 2.12** (pinned, flagged obsolete in [problems.txt](problems.txt)); the predictor is **XGBoost**; `morpheus_v1` is **pure PyTorch**.
- **~6 GB of committed binaries** (NASNet SavedModel duplicated in both branches, the 1 GB zip, multi-GB preprocessed-image temp dirs).
- **Self-acknowledged weaknesses** ([problems.txt](problems.txt)): *low cross-dataset generalizability, slow runtime, outdated TensorFlow*. The custom branch's own metrics confirm modest performance (test AUC ≈ **0.61**, F1 ≈ **0.32**).

### 1.4 Bottom line for the merge
This is the **trickiest of the three merges**, but **not because of plumbing** (that mirrors the vision repo) — it is because of a **stack and topology mismatch**:

1. **It is not one `nn.Module`.** The directive to "extract a pure `nn.Module`" maps cleanly **only to Stage 1**, and even there the weights are **TensorFlow**, not PyTorch. Stage 2 is a deterministic pandas transform; Stage 3 is a **gradient-boosted tree ensemble** that *cannot* be an `nn.Module` at all.
2. **The decisive engineering decision is the annotator's runtime:** keep TF as an optional backend, **export to ONNX**, or **re-host on PyTorch** (`timm` `nasnetalarge` + weight port). This single choice gates the whole merge (§6.1).
3. **The good news — data is already compatible.** The annotator consumes the **same loose per-embryo JPGs**, keyed by the **same `dish_well` identifier** (`D2013.02.19_S0675_I141_1`) used by `blasto_prediction`'s splits and `cellPIV`'s datasets. So split reconciliation is a **mapping problem, not a re-experiment**, exactly as in the vision plan.

Recommended framing: treat this not as "one model to port" but as **one literature *baseline pipeline*** registered in `morpheus_v1` as an inference-only comparator, with its three stages decomposed into (a) an annotator backend, (b) a feature transform, and (c) a tree-predictor wrapper.

---

## 2. Repository Map & Entry Points

### 2.1 Top-level layout
```
blasto_annotator_and_prediction/
├── README.md                          # one line, non-operational
├── problems.txt                       # known issues (generalizability, speed, TF obsolescence)
├── WORKSPACE_BRIEFING.md              # prior read-only inventory (factual background)
├── DB_Morpheus_withID.csv             # full clinical metadata + morphokinetic timings (6053 rows)
├── Normalized_sum_mean_mag_3Days_test.csv  # TEST-fold whitelist (656 rows): patient_id, dish_well, labels, age, 0–72h grid
├── _01_annotator_and_xgboost/         # BASELINE branch (2.1 GB)
└── _02_annotator_and_xgboost_custom/  # CUSTOM branch (4.1 GB)
```

### 2.2 Per-branch assets
| Asset | `_01` baseline | `_02` custom | Meaning |
|-------|:---:|:---:|---------|
| `blasto_annotator_and_pred.ipynb` | ✓ (19 MB, image outputs baked in) | ✓ (142 KB) | The only "source" |
| `NASNet_embryo_17class/` (SavedModel) | ✓ | ✓ | TF annotator, **duplicated** (~1.1 GB each) |
| `xgb_blastocyst_prediction.model` | ✓ | ✓ | XGBoost booster (primary target) |
| `xgb_good_blastocyst_prediction.model` | ✓ | ✓ | XGBoost booster |
| `xgb_poor_arrest_embryo_prediction.model` | ✓ | ✓ | XGBoost booster |
| `df_config.pkl` | ✓ | ✓ | `(3,2)` DataFrame: `target_label → list_var` (feature schema) |
| `Blastocyst_Prediction.zip` (+ `.part`) | ✓ (1 GB) | — | Re-downloadable weights bundle (gdown) |
| `sample_img_list.csv` / `.pkl` | ✓ | — | Demo image list, `/content/...` Colab paths |
| `sample_images*/` | ✓ | — | Demo frames + preprocessed |
| `temp_custom_preprocessed/` | — | ✓ (3.1 GB) | On-the-fly crop output (regenerated each run) |
| `blastocyst_prediction_metrics_summary.csv` / `_results.csv` / `model_comparison_line.csv` | — | ✓ | Bootstrap outputs |

### 2.3 Entry points (notebook cells, not CLI)
There is **no CLI and no `main()`**. The execution order is implicit in cell order:

| Cell (custom `_02`) | Action |
|---|---|
| Cell 3 | Config constants → gdown weights check → label dicts → define `convert_to_wide_format`, `process_df_data`, `parse_and_preprocess` → **run `parse_and_preprocess()`** (crop) → load NASNet → `ImageDataGenerator.flow_from_dataframe` → `argmax(nasnet.predict)` → wide pivot → `process_df_data` → `df_dataset` |
| Cell 4 | Load `df_config.pkl` + `xgb_blastocyst_prediction.model` → `Booster.predict(DMatrix)` → threshold 0.5 → classification report |
| Cell 5 | Extra scalar metrics (F1/BalAcc/MCC/Brier) |
| Cell 6 | **Bootstrap** (200 iters) → metrics CSV + compact comparison line + `bootstrap_distributions_Annotator_3Days.pkl` |
| Cells 7–8 | Sorting / a quoted-out `parse_and_preprocess_DEBUG` block (dead text) |

The baseline `_01` follows the same skeleton but its final cell (cell 12) **loops over all three `df_config` targets** instead of hardcoding one.

---

## 3. Model Architecture Mapping (the "literature model")

This is the core deliverable. The pipeline is **`raw frames → [Stage 1 CNN] → per-frame stage labels → [Stage 2 transform] → feature vector → [Stage 3 XGBoost] → outcome probability`**.

### 3.1 Stage 1 — `NASNet_embryo_17class` (frame annotator)
Verified by loading the SavedModel:

| Aspect | Value |
|--------|-------|
| Framework | **TensorFlow / Keras** SavedModel (`saved_model.pb` + `variables/` + `keras_metadata.pb`) |
| Backbone | **NASNetLarge** (inlined as a flat 1043-layer functional graph — `SeparableConv2D`×220, `BatchNormalization`×264, `Add`×110, `Concatenate`×26, etc.) |
| Input | `(None, 331, 331, 3)` — NASNetLarge's native 331² resolution |
| Channels | Frames are **grayscale**, **replicated to 3 channels** at load (`color_mode='rgb'`) |
| Normalization | `ImageDataGenerator(rescale=1/255)` — plain 0–1 scaling, **no ImageNet mean/std** |
| Output | `(None, 17)` softmax → `np.argmax` → integer class id |
| Params | **89,064,035** |
| Head | terminal `Dense(17)` named `dense_1` |

**The 17-class label space** (`annotation_label_dict`):
```
0:1cell      1:1cell_PN
2:2cell_G1   3:2cell_G2   4:2cell_G3
5:3cell_G1   6:3cell_G2   7:3cell_G3
8:4cell_G1   9:4cell_G2  10:4cell_G3
11:5-7cell_G1 12:5-7cell_G2 13:5-7cell_G3
14:8-cell_G1 15:8-cell_G2 16:8-cell_G3
```
i.e. **cell-stage × fragmentation-grade (G1/G2/G3)**, plus the two single-cell states. The grade groupings (`G1_label_list=[2,5,8,11,14]`, etc.) drive the downstream ratio features.

**Preprocessing that feeds Stage 1** (`parse_and_preprocess`, cell 3) — a classic OpenCV embryo-localization crop, run on-the-fly into a temp dir:
1. Grayscale + optional 98th-percentile level adjustment.
2. Multi-setting **Canny → morphological close/open → `findContours` → `fitEllipse`**, accepting the first ellipse with radius in `[min_size, max_size]`.
3. Crop a square around the ellipse center (+`extend_r=15` px), resize to **331²**, apply a **circular mask** (`bitwise_and`), write JPG.
4. **Fallback divergence:** `_02` falls back to a center crop (radius 250) if no contour is found; `_01` instead **drops** the frame (and also drops out-of-bounds crops). Settings differ too (`_02`: 5 settings, size `[140,650]`; `_01`: 4 settings, size `[150,600]`).

> Note: this is the **same family of crop** as `blasto_prediction`'s HDF5 builder, but the output geometry differs (**331² + circular mask** here vs the vision repo's resolution). The source frames are identical (§4).

### 3.2 Stage 2 — morphokinetic feature engineering (pandas)
Deterministic, no learned parameters:

1. **Round** each frame's `time(hpi)` to the nearest 0.25h; drop duplicate `(ID, time)`.
2. **`convert_to_wide_format`** — pivot to one row per embryo `ID`, columns = quarter-hour timepoints, values = annotation class id.
3. **`process_df_data`** — over the window `[TIMELAPSE_MIN, TIMELAPSE_MAX)` (24→64h baseline / 24→72h custom, 15-min step):
   - `ratio_1cell … ratio_8cell` = fraction of timepoints in each cell-stage group.
   - `ratio_G1/G2/G3` = fraction of 2–8-cell timepoints at each fragmentation grade.
   - Intermediate per-stage grade ratios + `argmax` → `grade_Ncell`, used to **(baseline only)** *replace/unify* the raw class ids in the time columns into grade-collapsed ids, then dropped.

The final per-embryo feature vector = **selected quarter-hour annotation columns + `age` + the 9 ratio features**.

### 3.3 Stage 3 — XGBoost predictors + `df_config` schema
`df_config.pkl` is a `(3,2)` DataFrame mapping each target to its **exact ordered feature list** (`list_var`):

| `target_label` | # features | Composition |
|----------------|:---:|-------------|
| `blastocyst_prediction` | **88** | 78 quarter-hour time columns (24.0–64.0h, sparse) + `age` + 9 ratios |
| `good_blastocyst_prediction` | **156** | 146 dense time columns + `age` + 9 ratios |
| `poor_arrest_embryo_prediction` | **76** | 66 time columns + `age` + 9 ratios |

Each target loads `xgb_<target>.model` as a native `xgb.Booster`, builds `DMatrix(df.reindex(columns=list_var).astype(float))`, and `predict()` returns the positive-class probability directly. The custom branch thresholds at **0.5** for `pred_class`. The baseline branch writes all three probabilities into one result table.

**Key schema contract:** the feature columns are **float-named** (timepoints are `24.0`, `25.25`, …). The notebook reconstructs them with a brittle `float(f) if str(f).replace('.','',1).isdigit() else f` parse — this must become an explicit, typed schema in the merge.

### 3.4 ⚠️ Correctness divergence between branches (must resolve before merge)
`process_df_data` differs in a way that **changes the XGBoost input distribution**:
- **`_01` baseline** performs **grade-unification**: it `.replace(...)`s the raw class ids in the time-grid columns with grade-collapsed ids (e.g. `grade_2cell==1 → {2,3,4}→4`) *before* dropping helper columns. The committed XGBoost models were presumably **trained on these unified labels**.
- **`_02` custom** computes the grades but **never applies the replacement** — it leaves the raw 17-class ids in the time columns.

If the boosters expect unified labels, the custom branch feeds them an out-of-distribution feature grid — a plausible contributor to the weak custom-branch AUC (~0.61). **Action:** determine which representation the boosters were trained on, make it canonical, and port exactly one feature-engineering implementation.

---

## 4. Data Layer Compatibility

### 4.1 How this model expects its data (today)
**Raw images, not features and not video tensors.** Concretely:
- A two-class directory tree of **loose per-embryo JPG folders**:
  `…/CorsoData/blastocisti/{blasto,no_blasto}/<dish_well>/<dish_well>_<idx>_<n>_<time>h.jpg`
  (confirmed: e.g. `D2013.02.19_S0675_I141_1/D2013.02.19_S0675_I141_1_100_0_24.8h.jpg`).
- Time is parsed from the filename via regex `_(\d+\.?\d*)h\.jpg$`.
- A **whitelist CSV** (`Normalized_sum_mean_mag_3Days_test.csv`) supplies the **set of valid `dish_well` IDs** and a `maternal age` map; folders whose `dish_well` is not in the CSV are skipped.
- Frames are cropped on-the-fly into a temp dir, then streamed by `ImageDataGenerator.flow_from_dataframe`.

So the input requirement is: **raw grayscale embryo frames + a per-embryo age scalar + an ID whitelist**. Features are **derived internally**; the model does *not* accept pre-extracted features as input.

### 4.2 The crucial compatibility fact: identical identity keys & source frames
This pipeline and `blasto_prediction` **read the same physical dataset**:

| Dimension | This repo (annotator) | `blasto_prediction` (vision) |
|-----------|----------------------|------------------------------|
| Source frames | loose JPGs under `blastocisti/{blasto,no_blasto}/<dish_well>/` | **same** loose JPGs, ingested into HDF5 |
| Grouping key | `dish_well` (`D…_S…_I###_<well>`) | `dish_well` (identical format, used in `splits_rome.txt`) |
| Label | folder (`blasto`=1 / `no_blasto`=0) | `labels` dataset in HDF5 |
| Storage | **none** — re-crops to temp each run | **monolithic HDF5** with baked `split`, `dish_well`, normalization attrs |
| Split source | a TEST-only whitelist CSV | a static `splits_rome.txt`, baked into HDF5 |

The annotator currently has **no HDF5 layer at all** — it does the slow crop every run (the notebook estimates ~2.5h). But because the `dish_well` keys match exactly, the annotator can be **re-pointed at the vision repo's HDF5 frames** (or a shared crop cache) instead of re-cropping, *provided the geometry is harmonized*: NASNet needs **331² + circular mask**; the vision models use a different size/normalization. The crop is the same operation; only the output transform differs.

### 4.3 Reconciling with the central patient-level CSV + `.env` (the target convention)
`morpheus_v1` resolves splits at runtime from a **central, patient-level stratified CSV** and reads all roots from **`.env`** (pattern exemplified by `cellPIV/new_project/.env.example` and the central `cellPIV/datasets/splits.txt` / `splits_2PN.txt`).

The annotator's `Normalized_sum_mean_mag_3Days_test.csv` is **functionally a single fold of that split** (it lists only the test `dish_well`s + age). Reconciliation:

| Concern | Today | Target action |
|---------|-------|---------------|
| **Split source of truth** | bespoke test-whitelist CSV | **Replace** with the central CSV; select `split == "test"` rows. The whitelist becomes a *derived view*, not an input. |
| **Grouping unit** | `dish_well` | Already matches the central split's key — **same dish-grouping** as the vision plan (§4.3 there). Confirm dish==patient; if a patient owns >1 dish, regroup upstream (shared concern across all three repos). |
| **`age` feature** | read from the whitelist CSV | Source `maternal age` from the **same central CSV / `DB_Morpheus_withID.csv`** so all models share one demographic table. |
| **Paths** | `BASE_PATH="/home/phd2/Scrivania"`, absolute CSV path, gdown `FILE_ID` | Route `DATA_ROOT`, `BLASTO_DIR`, `NO_BLASTO_DIR`, `SPLIT_CSV`, `ANNOTATOR_WEIGHTS`, `XGB_MODEL_DIR`, `OUTPUT_ROOT` through `.env`. |
| **Input medium** | re-crop loose JPGs each run | Consume the **shared HDF5 / crop cache** (Strategy A below) keyed by `dish_well`; resize to 331² + circular mask at load time. |

**Recommended (Strategy A — single source of truth):** the central CSV defines `dish_well → patient_id → split`. A shared loader yields, per embryo, the **time-ordered 24→72h frame stack** (from HDF5 or crop cache) + `age`, already filtered to the requested fold. The annotator stage then runs once and caches per-frame class ids. This guarantees the literature baseline is evaluated on **exactly the same patients in the same test fold** as ViT/ViViT/ROCKET/etc. — a precondition for the paper's paired-bootstrap comparisons (the `bootstrap_paired.py` machinery already lives in the vision repo and should be shared).

**Cheaper (Strategy B — adapter):** keep the annotator reading loose JPGs, but replace the whitelist CSV with a runtime filter derived from the central split. Leaves the slow per-run crop in place; acceptable as an interim step.

---

## 5. Architectural Cleanliness Assessment

| Concern | Evidence | Severity |
|---------|----------|:---:|
| **No `.py` / no package** | Entire logic in two `.ipynb`; no importable surface, no tests | High |
| **Hardcoded paths** | `BASE_PATH="/home/phd2/Scrivania"`, absolute `PATH_METADATA_CSV`, `TEMP_PREPROCESS_DIR="./temp_custom_preprocessed"` | High |
| **Hardcoded external dependency** | gdown `FILE_ID='15Y_R_rNcRym61EPp_jAKqHxkNuXEHX7y'` for a ~1 GB weights zip; brittle, unversioned, no checksum | High |
| **Mutable global blackboard** | `df_files`, `df_dataset`, `df_wide` mutated in place and shared across cells; module-level label lists (`G1_label_list`, …) | High |
| **Tight stage coupling** | crop + TF inference + pandas features + XGBoost + metrics + bootstrap all interleaved in single cells; no stage boundaries | High |
| **Stack obsolescence** | TF 2.12 pinned; `%pip install tensorflow==2.12`; flagged in `problems.txt`; collides with PyTorch target | High |
| **Branch divergence** | `_01` vs `_02` differ in window (64h/72h), target selection, **and grade-unification** (§3.4) | High (correctness) |
| **Hardcoded target** | `target_label = 'blastocyst_prediction'` in `_02` | Medium |
| **Brittle parsing** | float-vs-string feature reconstruction via `str(f).replace('.','',1).isdigit()`; regex-coupled to filename format | Medium |
| **Bare `except`** | level-adjustment and AUC blocks swallow all exceptions | Medium |
| **Append-mode output** | `model_comparison_line.csv` opened in `'a'` → duplicate rows across runs (already observed) | Low |
| **Committed heavy binaries** | ~6 GB: NASNet SavedModel ×2, 1 GB zip + `.part`, multi-GB temp crops, result CSVs | Medium (repo hygiene) |
| **No provenance** | no model card, no training manifest, no checksum, no seed discipline (seeds only in bootstrap loop) | Medium |
| **Dead code / cruft** | quoted-out `parse_and_preprocess_DEBUG`, baked-in image outputs in the 19 MB `_01` notebook | Low |

**Net:** the model *math and data semantics are sound and portable*; the *plumbing, packaging, and stack are not* — the same verdict as the vision repo, but with an added **TF→PyTorch** axis and an added **non-neural (tree) stage**.

---

## 6. Critical Assessment — What Must Be Rewritten

Ranked by importance for the merge.

### 6.1 Decide the annotator runtime (the gating decision)
The annotator is the only `nn.Module`-shaped component, but its weights are TensorFlow. Three options:

| Option | What | Pros | Cons | Recommendation |
|--------|------|------|------|----------------|
| **A. ONNX export** | `tf2onnx` → `model.onnx`; run via `onnxruntime`, or wrap with `onnx2torch` to expose an `nn.Module`-like forward | Framework-neutral; drops the TF dependency at inference; one portable artifact | One-time export + numerical-parity check; NASNet has many ops to validate | **Preferred** — best fit for a PyTorch release that must stay reproducible |
| **B. PyTorch re-host** | Build `timm`/`pretrainedmodels` `nasnetalarge`, port the 17-class head + weights | Native `nn.Module`, fully in-stack | TF→PyTorch NASNet weight transfer is **painful** (separable-conv ordering, asymmetric padding, BN epsilon); high risk of silent drift | Only if a validated `nasnetalarge` port already exists |
| **C. Keep TF as optional backend** | Isolate behind an `EmbryoStageAnnotator` interface; TF stays an extra | Zero porting | Keeps TF 2.12 alive in a PyTorch repo — contradicts the release goal | Interim fallback only |

Whichever is chosen, wrap it behind a **stable interface** `EmbryoStageAnnotator.annotate(frames) -> class_ids` so the rest of the pipeline is backend-agnostic. **Pin a numerical-parity test** (same frames → same argmax as the current TF model) as the acceptance gate.

### 6.2 Decompose the pipeline into three clean, testable units
Replace the notebook with three importable components (no globals, no I/O at import):
1. **`EmbryoStageAnnotator(nn.Module`-like wrapper)** — §6.1; constructor takes weights path (from `.env`); `forward/annotate` returns per-frame class ids; the 17-class label map is a typed constant.
2. **`MorphokineticFeatureBuilder`** — a pure transform: `(annotations_long, age) -> feature_row`; **one** implementation of `convert_to_wide_format` + `process_df_data` with the grade-unification question (§3.4) **resolved**; the time window is a parameter (24→72h). Unit-testable on synthetic annotation sequences.
3. **`XGBoostBlastoPredictor`** — a thin wrapper over `xgb.Booster` (**stays a tree model, not an `nn.Module`**): loads `df_config` + `xgb_<target>.model`, validates the feature schema, returns probability. Optionally export to ONNX for a TF/torch-free inference path.

These compose into a `LiteratureBaselinePipeline` registered in the model registry as an **inference-only comparator** alongside the trainable nets.

### 6.3 Kill hardcoded paths & the gdown dependency
All roots → `.env` (consumed by the clean config system). Replace the gdown auto-download with a **versioned, checksummed weights artifact** (or an explicit, documented fetch step). Remove `BASE_PATH`, the absolute CSV path, and the `temp_custom_preprocessed` literal.

### 6.4 Unify the data layer & reconcile the split
Drive everything from the **central patient-level CSV** (Strategy A, §4.3): one loader produces per-embryo frame stacks + `age` filtered to the requested fold, keyed by `dish_well`. Eliminate the bespoke whitelist CSV (make it a derived view). Reuse the shared HDF5 / crop cache rather than re-cropping each run. Share `bootstrap_paired.py` so the baseline is bootstrapped against the same test patients as every other model.

### 6.5 Consolidate evaluation & reporting
Fold the custom branch's bootstrap (AUC/F1/Acc/BalAcc/MCC/Brier with 95% CIs) into the **shared evaluation layer** the vision plan already specifies — same metric set, same `model_comparison_line.csv` schema (the custom branch deliberately mimics the "ViViT-style" line). Fix the append-mode duplicate-row bug; add run metadata (git hash, fold, timestamp).

### 6.6 Repository hygiene
Do **not** carry the ~6 GB of binaries into `morpheus_v1`. Externalize weights (annotator + 3 boosters) to a tracked artifact store; drop the 1 GB zip, the `.part`, the temp crop dirs, the baked-notebook image outputs, and the `sample_images*` demo.

**Portable as-is:** the 17-class label scheme, the ratio-feature definitions, the `df_config` per-target schema, the metric set, the bootstrap procedure. **Must change:** the runtime (TF), the packaging, the paths, and the (one) feature-engineering divergence.

---

## 7. Step-by-Step Merge Plan

> Assumes `morpheus_v1` provides: a `.env`+typed-config system, a `data/` layer with a central-CSV split resolver + shared frame source (HDF5 / crop cache), a `models/` registry, a shared `evaluate/`+`bootstrap_paired` layer, and an artifact store for weights.

### Phase 0 — Pre-flight & decisions
1. Confirm `morpheus_v1` package layout, config object, model-registry interface, and how non-neural / inference-only comparators are registered.
2. **Resolve §3.4:** determine whether the committed boosters were trained on **grade-unified** (`_01`) or **raw** (`_02`) time-grid labels; pick the canonical `process_df_data`. (Sanity check: run both feature variants through `xgb_blastocyst_prediction.model` on the test fold and compare to the recorded AUC ≈ 0.61.)
3. **Decide the annotator runtime** (§6.1) — recommend ONNX (Option A). Define the numerical-parity acceptance test.
4. **Resolve the split** — obtain the central CSV; define `dish_well → patient_id → split`; confirm the annotator's `dish_well`s are a subset of (and consistent with) the central test fold.

### Phase 1 — Configuration & paths
5. Add `.env` keys: `DATA_ROOT`, `BLASTO_DIR`, `NO_BLASTO_DIR`, `SPLIT_CSV`, `METADATA_CSV`, `ANNOTATOR_WEIGHTS`, `XGB_MODEL_DIR`, `OUTPUT_ROOT`, `ANNOTATOR_INPUT_SIZE=331`, `TIMELAPSE_MIN=24`, `TIMELAPSE_MAX=72`.
6. Delete `BASE_PATH`, the absolute CSV path, the gdown `FILE_ID`, and the `temp_custom_preprocessed` literal. Replace gdown with a checksummed artifact fetch.

### Phase 2 — Extract & validate the annotator (Stage 1)
7. Export `NASNet_embryo_17class` → ONNX (or port to `nasnetalarge`); wrap as `EmbryoStageAnnotator` behind a backend-agnostic interface; weights from `.env`.
8. **Parity gate:** assert identical `argmax` on a fixed frame sample vs the current TF SavedModel before proceeding.
9. Move the OpenCV crop (ellipse + circular mask + 331² resize, with the `_02` center-crop fallback) into a shared `preprocessing/` transform; harmonize with the vision crop so both consume the **same source frames**.

### Phase 3 — Feature builder (Stage 2)
10. Implement one `MorphokineticFeatureBuilder` (resolved §3.4 logic); window from config; emit the exact `df_config` feature schema. Add **unit tests** on synthetic annotation sequences (ratios sum correctly, grade logic, NaN handling).

### Phase 4 — Predictor (Stage 3)
11. Implement `XGBoostBlastoPredictor`: load `df_config.pkl` + `xgb_<target>.model`; **validate** the feature vector against `list_var` (typed schema, no `isdigit()` hack); return probability; support all three targets (restore the baseline multi-target loop). Optionally export boosters to ONNX.
12. Compose `LiteratureBaselinePipeline` (annotator → feature builder → predictor); register as an inference-only comparator in the model registry.

### Phase 5 — Data layer & split reconciliation
13. Implement the shared per-embryo loader (frames + `age`) driven by the central CSV (Strategy A); key by `dish_well`; serve the requested fold; cache per-frame annotations to avoid the ~2.5h re-crop.
14. Verify fold parity: the baseline's test patients == the central test fold == the vision/time-series test fold.

### Phase 6 — Evaluation & reporting
15. Run the pipeline through the shared `evaluate/` + `bootstrap_paired` layer; emit the standard metric set + `model_comparison_line.csv` line (`AutoAnnotator_XGBoost`, `Time=72`) with run metadata; fix the append-dedup bug.
16. Add a **model card** (provenance, training data, the literature citation, weight checksums) to close the governance gap.

### Phase 7 — Hygiene
17. Externalize all weights; exclude binaries/temp dirs/zip/sample images from the merged tree; strip baked notebook outputs.

---

## 8. Risks & Open Questions

| # | Risk / question | Why it matters | Mitigation |
|---|-----------------|----------------|-----------|
| R1 | **TF→PyTorch annotator parity** | NASNet ops (separable conv, padding, BN) can drift silently on export/port | Mandatory argmax-parity test (Phase 2.8); prefer ONNX over hand-port |
| R2 | **Grade-unification divergence (§3.4)** | Wrong feature representation feeds the boosters → invalid baseline | Determine the boosters' training representation before porting; one canonical builder |
| R3 | **Split parity across models** | Paired bootstrap is invalid if folds differ | Strategy A single source of truth; explicit fold-equality check |
| R4 | **dish vs patient grouping** | A patient with >1 dish could leak across folds (shared concern with vision/TS) | Confirm `dish == patient`; regroup upstream if not |
| R5 | **Provenance unknown** | No model card / training manifest / checksum | Add model card + checksums on merge |
| R6 | **gdown weights are unversioned** | Reproducibility depends on a live Drive link | Move to checksummed artifact store |
| R7 | **Runtime cost (~2.5h re-crop)** | Slow, flagged in `problems.txt` | Cache crops + per-frame annotations keyed by `dish_well` |
| R8 | **Weak baseline metrics (AUC≈0.61)** | May reflect the §3.4 bug *or* genuine model limits | Resolve R2 first, then re-evaluate; report honestly as the literature comparator |

---

## 9. Summary

| Question (from the brief) | Answer |
|---------------------------|--------|
| **What is it?** | A literature-based **3-stage hybrid**: TF/Keras NASNetLarge frame annotator (17 classes) → pandas morphokinetic features → XGBoost outcome predictor. Notebook-only, two divergent branches. |
| **What data does it expect?** | **Raw grayscale embryo frames** (loose per-`dish_well` JPGs) + a maternal-age scalar + an ID whitelist; features are derived internally. **Not** pre-extracted features, **not** video tensors. |
| **vs the HDF5 strategy?** | It has **no HDF5** — it re-crops loose JPGs each run. But it reads the **same source frames** keyed by the **same `dish_well`** as `blasto_prediction`'s HDF5, so it can be re-pointed at the shared frame source; the test-whitelist CSV is functionally one **fold** of the central split. |
| **How clean?** | Below the vision repo's bar: no package, hardcoded paths + gdown weights, mutable global DataFrames, fully interleaved stages, obsolete pinned TF, ~6 GB committed binaries, and a real feature-engineering divergence between branches. |
| **Merge strategy?** | Decompose into `EmbryoStageAnnotator` (ONNX-export the only `nn.Module`-shaped stage), `MorphokineticFeatureBuilder` (pure transform), and `XGBoostBlastoPredictor` (tree wrapper, stays non-neural); drive paths via `.env`; reconcile to the central patient-level CSV via `dish_well`; register as an **inference-only literature comparator** evaluated through the shared bootstrap layer. |

**One-line bottom line:** the literature model is **portable in substance but not in form** — its data identity already aligns with the unified repo (same `dish_well`, same frames), but it must be **decomposed into three clean stages** and its **annotator lifted off TensorFlow** (ONNX) before it can live as a first-class, reproducibly-evaluated baseline inside `morpheus_v1`.
