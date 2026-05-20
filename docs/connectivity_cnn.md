# Connectivity CNN Experiment

This is a separate experiment runner for testing a matrix-native CNN on
subject-level Fisher-z connectivity matrices. It does not use flattened edge
vectors, PCA, RFE, RFECV, or resized connectivity images.

The current runner uses a deeper residual-style 2D CNN, not the original
three-block shallow baseline. It trains directly on each subject's native
`ROI x ROI` connectivity matrix.

The script lives at [connectivity_cnn.py](/C:/Users/bibe/Downloads/XAI-for-ASD/connectivity_cnn.py).

## Server commands

Single-site AAL run:

```bash
python3 -u connectivity_cnn.py \
  --pipelines dparsf \
  --preprocessing_condition filt_global \
  --roi_atlas rois_aal \
  --site NYU \
  --dropout 0.5 \
  --learning_rate 1e-3 \
  --weight_decay 1e-2 \
  --batch_size 16 \
  --epochs 400 \
  --patience 75 \
  --verbose true
```

Pooled all-sites AAL run:

```bash
python3 -u connectivity_cnn.py \
  --pipelines dparsf \
  --preprocessing_condition filt_global \
  --roi_atlas rois_aal \
  --run_pooled_all_sites \
  --dropout 0.5 \
  --learning_rate 1e-3 \
  --weight_decay 1e-2 \
  --batch_size 16 \
  --epochs 400 \
  --patience 75 \
  --verbose true
```

All eligible sites comparison:

```bash
python3 -u connectivity_cnn.py \
  --pipelines dparsf \
  --preprocessing_condition filt_global \
  --roi_atlas rois_aal \
  --run_all_large_sites \
  --min_site_subjects 40 \
  --dropout 0.5 \
  --learning_rate 1e-3 \
  --weight_decay 1e-2 \
  --batch_size 16 \
  --epochs 400 \
  --patience 75 \
  --verbose true
```

You can combine the two experiment modes in one invocation:

```bash
python3 -u connectivity_cnn.py \
  --pipelines dparsf \
  --preprocessing_condition filt_global \
  --roi_atlas rois_aal \
  --run_all_large_sites \
  --run_pooled_all_sites \
  --verbose true
```

## Outputs

Per cohort run:

- `summary.json`
- `fold_metrics.csv`

Cross-site comparison:

- `<pipeline>_<atlas>_site_comparison.csv`
- `<pipeline>_<atlas>_site_comparison.json`
- `<pipeline>_<atlas>_experiment_summary.json`

All outputs are written under the chosen `--artifact_root`.
