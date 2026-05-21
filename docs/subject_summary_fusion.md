# Subject-Summary Fusion LOSO

`subject_summary_fusion.py` is a standalone experiment runner that reads each subject's raw `rois_aal .1D` file directly, builds deterministic subject summaries, and evaluates with a leave-one-site-out outer loop.

## What It Uses

- Input: `abide/downloads/Outputs/<pipeline>/<preprocessing_condition>/<roi_atlas>/*.1D`
- Default target: `dparsf` + `filt_global` + `rois_aal`
- Outer evaluation: full LOSO sweep over all eligible sites
- Optional single-site holdout mode with `--held_out_site <SITE_ID>`
- Inner train-only artifacts per LOSO round:
  - structured-feature scaler
  - ASD/control prototypes
  - TF-IDF vocabulary
  - configurable classifier backend

## Classifier Backends

- `rbf_svm`: stronger nonlinear classifier over the fused summary features
- `logistic_elasticnet`: the earlier linear baseline

## Server Run

```bash
cd ~/XAI-for-ASD-fresh
source ~/XAI-for-ASD/.venv/bin/activate
mkdir -p run_logs

nohup python3 -u subject_summary_fusion.py \
  --pipelines dparsf \
  --preprocessing_condition filt_global \
  --roi_atlas rois_aal \
  --artifact_root artifacts/subject_summary_fusion \
  --min_site_subjects 40 \
  --validation_size 0.2 \
  --classifier_type rbf_svm \
  --verbose true \
  > run_logs/subject_summary_fusion_dparsf_aal_loso.log 2>&1 &
```

## Single Held-Out Site Run

```bash
nohup python3 -u subject_summary_fusion.py \
  --pipelines dparsf \
  --preprocessing_condition filt_global \
  --roi_atlas rois_aal \
  --artifact_root artifacts/subject_summary_fusion \
  --min_site_subjects 40 \
  --validation_size 0.2 \
  --classifier_type rbf_svm \
  --held_out_site NYU \
  --verbose true \
  > run_logs/subject_summary_fusion_dparsf_aal_heldout_nyu.log 2>&1 &
```

Watch progress:

```bash
tail -f run_logs/subject_summary_fusion_dparsf_aal_loso.log
```

## Outputs

For the command above, outputs are written under:

```bash
artifacts/subject_summary_fusion/dparsf/filt_global/rois_aal/loso_all_sites/
```

Key files:

- `summary.json`
- `loso_site_metrics.csv`
- `subject_summaries.csv`
- `class_prototypes_by_site.json`
- `top_tokens_by_class_by_site.csv`

When you pass `--held_out_site NYU`, outputs are written under:

```bash
artifacts/subject_summary_fusion/dparsf/filt_global/rois_aal/held_out_NYU/
```
