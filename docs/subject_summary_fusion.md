# Subject-Summary Fusion LOSO

`subject_summary_fusion.py` is a standalone experiment runner that reads each subject's raw `rois_aal .1D` file directly, builds deterministic subject summaries, and evaluates with a leave-one-site-out outer loop.

## What It Uses

- Input: `abide/downloads/Outputs/<pipeline>/<preprocessing_condition>/<roi_atlas>/*.1D`
- Default target: `dparsf` + `filt_global` + `rois_aal`
- Outer evaluation: full LOSO sweep over all eligible sites
- Inner train-only artifacts per LOSO round:
  - structured-feature scaler
  - ASD/control prototypes
  - TF-IDF vocabulary
  - elastic-net logistic classifier

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
  --verbose true \
  > run_logs/subject_summary_fusion_dparsf_aal_loso.log 2>&1 &
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
