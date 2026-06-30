# WORKSPACE_BRIEFING

## 1) Scope, method, and constraints

- Repository root inspected: `/home/phd2/Scrivania/CorsoRepo/blasto_annotator_and_prediction`.
- Investigation mode: read-only.
- No code logic was edited.
- No model training was run.
- No long evaluations were executed.
- Only lightweight file inspection and metadata extraction were used.
- Main objective: describe current workspace state with factual evidence.
- Secondary objective: identify immediate reproducibility and maintenance gaps.
- Language target: English.
- Evidence sources were limited to files currently present in this workspace.
- If a detail cannot be proven from files, it is marked as unknown.
- Unknown format used in this document: `UNKNOWN — reason: ...`.
- The workspace appears notebook-centric, not package-centric.
- There is no standalone Python source module tree in this repository.
- Existing outputs suggest prior successful runs happened inside notebooks.
- The repository contains both baseline and custom workflow variants.
- The custom variant contains explicit bootstrap metric export logic.
- The baseline variant contains multi-target prediction loops.
- The root README is minimal and not operationally complete.
- There is an explicit root `problems.txt` listing known project issues.
- Artifact storage is heavy relative to source-doc files.
- Most storage is consumed by model folders and generated/preprocessed image assets.
- Important operational detail: large generated folders are present in-repo.
- This has direct impact on portability and clone/update cost.
- The project currently depends on binary model artifacts committed in the tree.
- Model inference chain appears to be:
  - preprocessed image frames -> NASNet class labels
  - label-time table -> engineered time/rate features
  - engineered features -> XGBoost probability/class outputs
- `df_config.pkl` is central for mapping target labels to feature subsets.
- The same `df_config` structure is present in both `_01` and `_02` branches.
- The custom branch adds report files and compact model comparison CSV line output.
- Notebook imports indicate mixed dependency strata:
  - deep learning (TensorFlow/Keras)
  - classic ML (xgboost, sklearn)
  - image processing (opencv)
  - plotting (matplotlib/seaborn)
  - utility I/O (gdown, pickle, zipfile)
- There is no lockfile or explicit environment manifest in repository root.
- Environment reproducibility therefore currently depends on out-of-band setup.
- Pathing in sample CSV includes `/content/...` style entries from notebook context.
- This indicates the notebooks were likely designed for Colab-like paths initially.
- Local execution path patches exist in custom notebook logic.
- Some notebook output cells contain traceback remnants from past runs.
- Those remnants are useful as historical evidence but not active runtime state.
- Current briefing is bounded to what is directly observable.

Evidence snippet (root issues):

Source: `problems.txt` lines 1-3

```text
Very low generalizability across datasets
Very slow execution time
Based on an outdated TensorFlow version, causing compatibility issues with newer releases
```

Evidence snippet (root README):

Source: `README.md` line 1

```markdown
# blasto_annotator_and_prediction
```

---

## 2) Repository layout and storage profile

- Top-level folders:
  - `_01_annotator_and_xgboost/`
  - `_02_annotator_and_xgboost_custom/`
- Top-level files:
  - `README.md`
  - `problems.txt`
  - `DB_Morpheus_withID.csv`
  - `Normalized_sum_mean_mag_3Days_test.csv`
  - `.gitignore`
- Root size profile (human-readable):
  - `README.md`: 4.0K
  - `problems.txt`: 4.0K
  - `DB_Morpheus_withID.csv`: 2.0M
  - `Normalized_sum_mean_mag_3Days_test.csv`: 3.6M
  - `_01_annotator_and_xgboost`: 2.1G
  - `_02_annotator_and_xgboost_custom`: 4.1G
- Combined major folders exceed 6G.
- Heavy size concentration appears in model and generated image folders.
- Key large subfolders:
  - `_01_.../sample_images`: 30M
  - `_01_.../sample_images_preprocessed`: 12M
  - `_01_.../NASNet_embryo_17class`: 1.1G
  - `_02_.../NASNet_embryo_17class`: 1.1G
  - `_02_.../temp_custom_preprocessed`: 3.1G
- The custom branch is roughly ~2G larger than baseline.
- Large binary artifacts are committed directly in workspace.
- `.gitignore` exists but currently does not show explicit CSV exclusions in active rules.
- `.gitignore` contains image extension ignores (`*.png`, `*.jpg`, etc.).
- Presence of image folders despite ignore rules means files were likely added before ignore, or exceptions/workflow differs.
- Root has `.git/` folder, so this is an active git repository.
- There is no `src/` package tree.
- There is no `tests/` directory found by filename search.
- There are no `.py` files at all in the workspace tree.
- Operational logic is therefore primarily embedded in notebooks.
- This increases review/debug complexity vs module-based projects.
- It also increases merge-conflict risk for collaborative updates.
- Artifact inventory (selected):
  - notebook files (`.ipynb`)
  - model files (`.model`)
  - tensorflow saved model files (`saved_model.pb`, `keras_metadata.pb`)
  - pickle config/data files (`.pkl`)
  - CSV datasets/results
  - one split archive part (`.part`)
- No `requirements.txt` at root.
- No `pyproject.toml` at root.
- No `environment.yml` at root.
- No `Pipfile` at root.
- No explicit container setup files were observed in root listing.
- Root documentation footprint is minimal relative to data/model footprint.
- The workspace appears to be an experiment snapshot repository rather than a packaged application.

Evidence snippet (top-level size summary):

Source: shell inventory output

```text
2.0M    ./DB_Morpheus_withID.csv
3.6M    ./Normalized_sum_mean_mag_3Days_test.csv
2.1G    ./_01_annotator_and_xgboost
4.1G    ./_02_annotator_and_xgboost_custom
```

Evidence snippet (`.gitignore` image rules):

Source: `.gitignore` lines 9-14

```ignore
*.png
*.jpg
*.jpeg
*.gif
*.tif
*.tiff
```

---

## 3) Workflow architecture and execution path

- Both notebook branches implement a similar high-level workflow skeleton.
- Stage A: frame-level input parsing and preprocessing.
- Stage B: NASNet model inference for multiclass embryo state annotation.
- Stage C: time-indexed annotation reshaping to wide-format feature table.
- Stage D: engineered ratio features over selected timelapse windows.
- Stage E: XGBoost predictor inference for target endpoints.
- Stage F: evaluation and export of metrics/results artifacts.
- In `_02` custom notebook, explicit constants exist for time windows.
- Timelapse bounds in custom notebook:
  - `TIMELAPSE_MIN = 24`
  - `TIMELAPSE_MAX = 72`
- Helper functions identified in custom notebook:
  - `convert_to_wide_format(...)`
  - `process_df_data(...)`
  - `parse_and_preprocess(...)`
- The same conceptual helper set appears in baseline notebook.
- Baseline notebook function defaults show `timelapse_max=64` inside one function signature.
- Custom notebook appears to push full run window to 72h in main flow.
- Path strategy in custom branch is local filesystem oriented.
- Path strategy in baseline sample CSV still points to `/content/sample_images/...`.
- This indicates mixed runtime assumptions across versions.
- Image ingestion in baseline uses `ImageDataGenerator(...).flow_from_dataframe(...)`.
- NASNet loading is done via `tf.keras.models.load_model(...)`.
- XGBoost loading is done with `xgb.Booster()` and `booster.load_model(...)`.
- Probability output is thresholded into classes in custom notebook.
- Custom notebook target selection currently hardcoded to blastocyst target in one section.
- Baseline notebook loops targets from `df_config` rows.
- Feature subset for each target is retrieved from `df_config['list_var']`.
- The project relies on model files named with `xgb_<target>.model` convention.
- The notebook cell ordering includes historical error output in `_02`.
- That output indicates at least one run attempted to use `df_dataset` before guaranteed availability.
- Custom notebook includes a debug/preprocessing code block embedded as a quoted block near end.
- That debug block is non-executed text in current state.
- Flow robustness depends heavily on notebook execution order discipline.
- There is no guard script enforcing cell order.
- There is no CLI wrapper for standardized end-to-end runs.
- There is no unit test harness around feature engineering helpers.
- There is no explicit schema validation layer for intermediate DataFrames.
- This raises silent-failure risk when inputs drift.
- Branch divergence summary:
  - `_01`: baseline, multi-target style output table.
  - `_02`: custom, explicit metrics summary + bootstrap exports.

Evidence snippet (custom timelapse constants + function entrypoints):

Source: `_02_annotator_and_xgboost_custom/blasto_annotator_and_pred.ipynb` lines 796-797, 838, 845, 894

```python
TIMELAPSE_MIN = 24
TIMELAPSE_MAX = 72
def convert_to_wide_format(df_long, time_name, label_name):
def process_df_data(df_data, timelapse_min=24, timelapse_max=64, interval=15):
def parse_and_preprocess():
```

Evidence snippet (baseline data generator + model load):

Source: `_01_annotator_and_xgboost/blasto_annotator_and_pred.ipynb` lines 785, 816

```python
pred_dataset = ImageDataGenerator(rescale=1.0/255).flow_from_dataframe(
model = tf.keras.models.load_model(model_path)
```

---

## 4) Data assets and schema observations

- Core root datasets:
  - `DB_Morpheus_withID.csv` (6054 lines total)
  - `Normalized_sum_mean_mag_3Days_test.csv` (657 lines total)
- `DB_Morpheus_withID.csv` appears to be metadata + developmental timings + labels.
- `Normalized_sum_mean_mag_3Days_test.csv` appears to be wide time-series magnitudes by quarter-hour.
- `Normalized...` header includes dense temporal columns from `0.00h` through `72.00h`.
- `Normalized...` includes target-relevant columns:
  - `BLASTO NY`
  - `eup_aneup`
  - `PN`
  - `maternal age`
- `DB_Morpheus_withID.csv` includes many clinical/embryology timing columns:
  - `tPNa`, `tPNf`, `t2`, `t3`, `t4`, ...
  - blastocyst stage timing markers (`tM`, `tSB`, `tB`, `tEB`, `t-biopsy`)
- `DB_Morpheus...` contains `dish_well` composite key and separate `dish`,`well`.
- Case formatting in label values is mixed (`euploide`, `Euploide`, `Aneuploide`, etc.).
- Mixed casing/typing implies normalization requirements for stable ML preprocessing.
- Some rows contain missing tokens and placeholder dashes in timing fields.
- Sample-level image list file exists in baseline branch:
  - `_01_annotator_and_xgboost/sample_img_list.csv`
- This list includes columns:
  - `ID`
  - `age`
  - `time(hpi)`
  - `img_path`
- Example image paths in sample list are `/content/sample_images/...` style.
- That pathing may be invalid on local Linux without remapping.
- Pickle data artifacts observed:
  - `_01_.../sample_img_list.pkl`
  - `_01_.../df_config.pkl`
  - `_02_.../df_config.pkl`
- Pickle schema checks:
  - `sample_img_list.pkl`: DataFrame `(322, 4)` with expected image-list columns.
  - both `df_config.pkl`: DataFrame `(3, 2)` with `target_label`, `list_var`.
- `df_config` appears to drive feature-column subsets per prediction target.
- Three configured targets exist in `df_config`:
  - `blastocyst_prediction`
  - `good_blastocyst_prediction`
  - `poor_arrest_embryo_prediction`
- Time-grid values in `list_var` include many quarter-hour markers plus ratio features.
- Ratio features in `list_var` include:
  - `ratio_1cell`
  - `ratio_2cell`
  - `ratio_3cell`
  - `ratio_4cell`
  - `ratio_5cell`
  - `ratio_8cell`
  - `ratio_G1`
  - `ratio_G2`
  - `ratio_G3`
- Label column cardinalities are not fully profiled in this briefing.
- `UNKNOWN — reason: full cardinality scan for all categorical columns was not run.`
- Missing-value percentages per column are not included here.
- `UNKNOWN — reason: full null audit was not run to keep execution lightweight.`
- Primary data-source-of-truth between root CSVs and notebook-generated tables is not fully documented.
- `UNKNOWN — reason: notebook narrative comments do not explicitly define authoritative source precedence.`
- Potential schema risk: mixed delimiter semantics in domain columns (text values, dashes, blanks).
- Potential integrity risk: duplicated or near-duplicated identifier semantics (`dish_well`, `ID`).

Evidence snippet (`DB_Morpheus_withID.csv` header):

Source: `DB_Morpheus_withID.csv` line 1

```csv
patient_id,dish,well,dish_well,maternal age,sperm quality,mezzo di coltura,PN,BLASTO NY,eup_aneup,LB,video_start,tPNa,tPNf,t2,t3,t4,t5,t6,t7,t8,t9+,tM,tSB,tB,tEB,t-biopsy,PN1a,PN2a,PN3a,altri Pna,PN1f,PN2f,PN3f,altri PNf,CP
```

Evidence snippet (`Normalized_sum_mean_mag_3Days_test.csv` header prefix):

Source: `Normalized_sum_mean_mag_3Days_test.csv` line 1

```csv
patient_id,dish_well,BLASTO NY,eup_aneup,PN,maternal age,0.00h,0.25h,0.50h,0.75h,...,72.00h
```

Evidence snippet (`sample_img_list.csv` shape indicator):

Source: `_01_annotator_and_xgboost/sample_img_list.csv` lines 1-3

```csv
ID,age,time(hpi),img_path
Test01,32,24.0,/content/sample_images/Test01_001.jpg
Test01,32,24.25,/content/sample_images/Test01_002.jpg
```

---

## 5) Model assets and target configuration

- TensorFlow NASNet assets are stored in both branches under similarly named folders.
- Each NASNet folder includes:
  - `saved_model.pb`
  - `keras_metadata.pb`
  - variables directory (`variables.index`, `variables.data-00000-of-00001`)
- XGBoost model files in each branch:
  - `xgb_blastocyst_prediction.model`
  - `xgb_good_blastocyst_prediction.model`
  - `xgb_poor_arrest_embryo_prediction.model`
- Naming convention is consistent between `_01` and `_02`.
- `df_config.pkl` in both branches lists same three target labels.
- Custom notebook includes hardcoded selection:
  - `target_label = 'blastocyst_prediction'` in one inference section.
- Baseline notebook appears to iterate through configured targets from config table.
- XGBoost loading style in notebooks:
  - initialize `xgb.Booster()`
  - call `load_model(xgb_path)`
  - build `xgb.DMatrix(X)` from feature columns
- Thresholding in custom notebook:
  - binary class assigned via `(prob_blasto > 0.5)`
- Model path resolution uses joined paths from model root + target label pattern.
- Model provenance metadata is not explicitly versioned in repository.
- `UNKNOWN — reason: no metadata manifest links model hashes to training runs.`
- Training data version IDs for each model are not present in repo docs.
- `UNKNOWN — reason: no model card or training log artifacts found.`
- There is no explicit model checksum validation step in notebook code snippets observed.
- There is no fallback behavior if one model file is missing.
- There is no runtime compatibility assertion for TensorFlow model format versions.
- Known risk from `problems.txt`: legacy TensorFlow compatibility concerns.
- This can affect portability on newer environments.
- Baseline branch includes a `.part` archive piece:
  - `Blastocyst_Prediction.zipmo3_87zh.part`
- Its assembly/extraction status is unclear from current files.
- `UNKNOWN — reason: no matching full archive or reconstruction instructions found.`
- Model duplication across branches increases repository size.
- It also increases update mismatch risk if only one branch receives model refresh.
- Suggested governance gap:
  - branch-level model version pinning is undocumented.

Evidence snippet (custom target + booster loading):

Source: `_02_annotator_and_xgboost_custom/blasto_annotator_and_pred.ipynb` lines 1144-1151

```python
target_label = 'blastocyst_prediction'
xgb_path = os.path.join(model_root, f'xgb_{target_label}.model')
booster = xgb.Booster()
booster.load_model(xgb_path)
```

Evidence snippet (`df_config.pkl` extracted targets):

Source: pickle inspection output

```text
blastocyst_prediction
good_blastocyst_prediction
poor_arrest_embryo_prediction
```

---

## 6) Prediction outputs, metrics, and bootstrap artifacts

- Custom branch output files include:
  - `blastocyst_prediction_metrics_summary.csv`
  - `blastocyst_prediction_results.csv`
  - `model_comparison_line.csv`
- `blastocyst_prediction_metrics_summary.csv` contains CI-like percentile summary.
- Columns in summary:
  - `Mean`
  - `2.5th Percentile`
  - `97.5th Percentile`
- Reported metrics include:
  - `AUC`
  - `F1-Score`
  - `Accuracy`
  - `Balanced Accuracy`
  - `MCC`
  - `Brier Score`
- Current observed means:
  - AUC: `0.61359877547612`
  - F1: `0.3210892661645889`
  - Accuracy: `0.5044274809160306`
  - Balanced Accuracy: `0.545409936427074`
  - MCC: `0.11995458563988773`
  - Brier: `0.3577302727103233`
- `model_comparison_line.csv` has repeated model-time entries.
- Two rows exist for `AutoAnnotator_XGBoost` at `Time=72` with different metric blocks.
- This suggests append behavior across multiple runs.
- Custom notebook contains bootstrap code with `n_iterations = 200`.
- Sampling uses sklearn `resample(... replace=True, random_state=i)`.
- Notebook defines `TIME_POINT = 72` for compact line exports.
- Notebook writes metrics summary to CSV and compact line to comparison CSV.
- Notebook includes pickle save call for bootstrap distributions:
  - `'bootstrap_distributions_Annotator_3Days.pkl'`
- That specific pickle file is not currently present in root/known branch listings.
- `UNKNOWN — reason: notebook includes save code, but file not found in current tree scan.`
- `blastocyst_prediction_results.csv` is large (764 lines total observed by reader metadata).
- It includes per-ID wide time columns and engineered ratio columns.
- It includes prediction outputs:
  - `prob_blasto`
  - `pred_class`
- Ground truth column present in output:
  - `GroundTruth`
- This file is suitable for post-hoc error analysis.
- There is no separate confusion matrix image artifact committed.
- `UNKNOWN — reason: no persisted plot outputs found in root-level scan.`
- There is no run manifest tying output files to notebook execution timestamp.
- `UNKNOWN — reason: output metadata fields for run ID are absent in CSV schema.`

Evidence snippet (metrics summary values):

Source: `_02_annotator_and_xgboost_custom/blastocyst_prediction_metrics_summary.csv` lines 1-7

```csv
,Mean,2.5th Percentile,97.5th Percentile
AUC,0.61359877547612,0.5700476723697412,0.660449298886458
F1-Score,0.3210892661645889,0.2631358618900769,0.37613346855983776
Accuracy,0.5044274809160306,0.4671374045801527,0.5389312977099237
Balanced Accuracy,0.545409936427074,0.5140328325759113,0.5712532141349669
MCC,0.11995458563988773,0.037854228017898305,0.18960491424542186
Brier Score,0.3577302727103233,0.334250033646822,0.3800621174275875
```

Evidence snippet (bootstrap code indicators):

Source: `_02_annotator_and_xgboost_custom/blasto_annotator_and_pred.ipynb` lines 1291, 1303, 1321, 1351-1352

```python
from sklearn.utils import resample
TIME_POINT = 72
df_boot = resample(df_dataset, replace=True, n_samples=n_samples, random_state=i)
with open('bootstrap_distributions_Annotator_3Days.pkl', 'wb') as f:
    pickle.dump(bootstrap_results, f)
```

---

## 7) Dependencies, runtime, and reproducibility posture

- Active configured Python environment observed during inspection:
  - type: `venv`
  - version: `3.11.14`
  - interpreter path: `/home/phd2/Scrivania/CorsoVenvs/annotatorVenv/bin/python`
- Notebook import inventory indicates required libraries include:
  - `numpy`
  - `pandas`
  - `opencv-python` (`cv2`)
  - `tensorflow`
  - `xgboost`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `Pillow`
  - `tqdm`
  - `gdown`
  - standard libs (`os`, `sys`, `re`, `zipfile`, `shutil`, `pickle`, etc.)
- Baseline notebook also imports:
  - `torch`
  - `NASNetLarge` symbol from Keras applications
  - `xgboost.XGBClassifier`
- A strict environment specification file is absent.
- `UNKNOWN — reason: no `requirements.txt`, `pyproject.toml`, or `environment.yml` found.`
- This prevents deterministic reconstruction from repository alone.
- Notebook cells include install/download style logic (`gdown` etc.).
- This implies hidden external dependencies and URLs beyond repo files.
- External resource stability is not guaranteed.
- `UNKNOWN — reason: external download links are not tracked with checksums in docs.`
- Mixed path assumptions (`/content/...` and local absolute paths) reduce portability.
- Execution order in notebooks is not enforced by tooling.
- There is no CI config or validation pipeline detected in root listing.
- There is no formal packaging metadata.
- There are no pinned CUDA/cuDNN compatibility notes.
- `UNKNOWN — reason: hardware/accelerator assumptions are not documented in repository.`
- There are no deterministic seed conventions documented for end-to-end runs.
- `UNKNOWN — reason: seeds appear partially set in bootstrap loops only.`
- There is no dataset version pinning statement.
- There is no data license statement in root docs.
- `UNKNOWN — reason: licensing terms for datasets/models are not defined in inspected files.`
- There is no model governance artifact (model cards, datasheets, changelog).
- Reproducibility maturity (current): low-to-moderate.

Evidence snippet (custom notebook imports):

Source: `_02_annotator_and_xgboost_custom/blasto_annotator_and_pred.ipynb` lines 768-777

```python
import numpy as np
import pandas as pd
import cv2
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import gdown
```

Evidence snippet (baseline notebook imports excerpt):

Source: `_01_annotator_and_xgboost/blasto_annotator_and_pred.ipynb` lines 131-141

```python
import tensorflow as tf
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.nasnet import NASNetLarge
import xgboost
import sklearn
import torch
from xgboost import XGBClassifier
import warnings
```

---

## 8) Operational risks and technical debt

- Risk 1: low cross-dataset generalizability explicitly acknowledged by maintainers.
- Risk 2: slow runtime explicitly acknowledged by maintainers.
- Risk 3: TensorFlow version obsolescence acknowledged in root issues file.
- Risk 4: notebook-only architecture increases fragility and lowers testability.
- Risk 5: absence of package manifests weakens reproducibility.
- Risk 6: heavy binary artifacts in git increase storage and clone overhead.
- Risk 7: duplicated model directories across branches may drift silently.
- Risk 8: mixed path conventions (`/content/...` vs local absolute) break portability.
- Risk 9: no formal data validation pipeline for CSV schema drift.
- Risk 10: no formal model metadata for provenance and rollback.
- Risk 11: no test suite detected for feature engineering correctness.
- Risk 12: no CI validation detected for notebook execution health.
- Risk 13: notebook execution order dependency can produce stale-variable errors.
- Risk 14: historical traceback outputs in notebook can obscure current status.
- Risk 15: missing explicit hardware requirements may cause runtime mismatches.
- Risk 16: external downloads via `gdown` are brittle without checksum validation.
- Risk 17: unknown licensing context for data/model reuse.
- Risk 18: potentially large generated folders committed with source may impede collaboration.
- Risk 19: no run metadata linked to output CSVs reduces auditability.
- Risk 20: no documented threshold calibration beyond fixed `0.5` in custom inference block.
- Risk 21: class imbalance handling strategy is not documented.
- `UNKNOWN — reason: no class-balance analysis artifact found in inspected files.`
- Risk 22: version mismatch between baseline and custom branches could produce inconsistent results.
- Risk 23: no explicit migration notes between `_01` and `_02` branches.
- Risk 24: no centralized constants/config module; constants are cell-local.
- Risk 25: weak discoverability for new contributors due sparse README.
- Risk 26: branch naming (`_01`, `_02`) is informative but lacks lifecycle policy docs.
- Risk 27: no data anonymization statement for clinical-like metadata fields.
- `UNKNOWN — reason: privacy/compliance posture is not documented in repository files.`
- Risk 28: no disaster recovery notes for lost external dependencies.
- Risk 29: no artifact pruning policy for generated intermediate image folders.
- Risk 30: no explicit semantic versioning for outputs.

Risk evidence snippet (maintainer issue list):

Source: `problems.txt` lines 1-3

```text
Very low generalizability across datasets
Very slow execution time
Based on an outdated TensorFlow version, causing compatibility issues with newer releases
```

---

## 9) Unknowns and unresolved questions

- Unknown 1: canonical entrypoint notebook (`_01` vs `_02`) for production use.
- `UNKNOWN — reason: no explicit recommendation in root documentation.`
- Unknown 2: authoritative data source among root CSVs vs notebook-generated datasets.
- `UNKNOWN — reason: no data lineage document found.`
- Unknown 3: exact training dataset and date for each committed model file.
- `UNKNOWN — reason: no training manifest/model card present.`
- Unknown 4: whether baseline and custom model files are identical weights.
- `UNKNOWN — reason: no hash comparison manifest committed.`
- Unknown 5: intended meaning and allowed values for some domain columns (`LB`, `CP`, etc.).
- `UNKNOWN — reason: no data dictionary found.`
- Unknown 6: expected minimum package versions for successful execution.
- `UNKNOWN — reason: no lockfile or dependency pinning file found.`
- Unknown 7: expected hardware profile (CPU-only vs GPU-accelerated runtime).
- `UNKNOWN — reason: no environment/hardware guidance found.`
- Unknown 8: whether notebook debug blocks are historical-only or expected operational tools.
- `UNKNOWN — reason: debug block exists as quoted text without accompanying docs.`
- Unknown 9: whether bootstrap distribution pickle is intentionally excluded or accidentally missing.
- `UNKNOWN — reason: save call exists in notebook but file absent in scanned artifacts.`
- Unknown 10: rationale for duplicated `AutoAnnotator_XGBoost,72` rows in comparison CSV.
- `UNKNOWN — reason: append policy is not explained in notebook comments/docs.`
- Unknown 11: whether root `.gitignore` strategy is intentionally broad VisualStudio template or incidental.
- `UNKNOWN — reason: no repository maintenance notes found.`
- Unknown 12: whether there is an external private dataset not represented here.
- `UNKNOWN — reason: references to download/setup are partial and not fully enumerated.`
- Unknown 13: whether clinical data governance constraints apply to this repository.
- `UNKNOWN — reason: no compliance documentation found.`
- Unknown 14: intended long-term branch strategy for `_01` and `_02`.
- `UNKNOWN — reason: no branch policy document found.`
- Unknown 15: whether metric thresholds are clinically validated or exploratory.
- `UNKNOWN — reason: no validation protocol document found.`
- Unknown 16: whether all notebook outputs are up-to-date with current committed code.
- `UNKNOWN — reason: output timestamps are not embedded in generated CSVs.`
- Unknown 17: whether failed notebook cell outputs represent current blockers.
- `UNKNOWN — reason: notebook execution order/history state is not authoritative from static file read.`
- Unknown 18: intended deployment target (research-only, local tool, service integration).
- `UNKNOWN — reason: README contains no deployment/use-case statement.`
- Unknown 19: expected user-facing API or command surface.
- `UNKNOWN — reason: no `.py` modules or CLI scripts found.`
- Unknown 20: whether root CSVs are sample subsets or full cohorts.
- `UNKNOWN — reason: no dataset scope note in docs.`

---

## 10) Immediate recommendations and prioritized next actions

- Priority A1: add a real root README with reproducible run steps.
- Priority A2: declare canonical notebook (`_02` seems richer for reporting).
- Priority A3: export dependencies to a pinned `requirements.txt`.
- Priority A4: add environment notes (Python version, TensorFlow compatibility).
- Priority A5: externalize hardcoded paths to a single config cell/file.
- Priority A6: document data dictionary for all key CSV columns.
- Priority A7: add model provenance manifest (hash, train date, train data tag).
- Priority A8: add output manifest with run ID + timestamp + git commit hash.
- Priority A9: decide policy for large generated folders in git.
- Priority A10: unify `_01` and `_02` or clearly mark one as deprecated.
- Priority B1: modularize feature engineering helpers into standalone `.py` file.
- Priority B2: add lightweight unit tests for feature ratios and mapping logic.
- Priority B3: add schema assertions before inference.
- Priority B4: validate missing-value handling strategy and class label normalization.
- Priority B5: implement deterministic seed block at notebook start.
- Priority B6: include confusion matrix/ROC artifact export with run metadata.
- Priority B7: add guardrails for notebook execution order.
- Priority B8: include explicit calibration/threshold discussion for `pred_class`.
- Priority B9: consolidate model files and avoid branch duplication drift.
- Priority B10: add CI check for notebook static quality (at least lint/schema checks).
- Priority C1: document privacy/compliance posture for clinical-like data fields.
- Priority C2: include license statement for code, data, and model artifacts.
- Priority C3: add reproducibility section in README with expected runtime budget.
- Priority C4: preserve compact result append semantics with explicit dedupe rules.
- Priority C5: add smoke-test subset to avoid full heavy runs during validation.
- Priority C6: include changelog entries for model and notebook updates.
- Priority C7: migrate repetitive notebook code to reusable functions/modules.
- Priority C8: add explicit branch naming semantics and maintenance policy.
- Priority C9: publish minimal `runbook.md` for new users.
- Priority C10: define archival policy for intermediate preprocessed image data.
- Suggested execution sequence:
  - Step 1: dependency pinning + README run path.
  - Step 2: config/path normalization.
  - Step 3: provenance + output metadata.
  - Step 4: modularization + tests.
  - Step 5: repository hygiene for heavy artifacts.
- Suggested measurable acceptance criteria:
  - fresh clone to successful `_02` inference in one documented command sequence.
  - metrics CSV regeneration reproducible within expected variation bounds.
  - no manual path edits required by operator.
  - run output includes machine-readable run metadata.
  - branch ownership/canonical status documented.

Final state summary:

- Workspace is rich in artifacts and practical notebook logic.
- Workspace is weak in reproducibility packaging and governance metadata.
- Custom branch contains useful evaluation reporting extensions.
- Immediate wins are mostly documentation/config/provenance hardening.
