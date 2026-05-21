# 3D NIfTI Autoencoder

This standalone runner trains an unsupervised 3D convolutional autoencoder on
preprocessed ABIDE `func_preproc.nii.gz` volumes. It lazily reads one subject at
a time from a manifest CSV, collapses the 4D time axis into a single 3D volume,
and exports a latent vector per subject for downstream classification.

## Expected Inputs

- A manifest CSV with at least:
  - `file_id`
  - `site_id`
  - `label`
  - `nifti_path`
- Directly readable `.nii.gz` files. They should stay compressed.

## NYU Pilot Run on `soair`

```bash
cd /home/soair/XAI-for-ASD
source .venv/bin/activate
mkdir -p run_logs

nohup python3 -u nifti_3d_autoencoder.py \
  --manifest artifacts/nifti_manifest_dparsf_nyu.csv \
  --artifact_root artifacts/nifti_3d_autoencoder/nyu_variance_latent1024 \
  --collapse_mode variance \
  --batch_size 2 \
  --epochs 100 \
  --patience 15 \
  --min_delta 1e-4 \
  --learning_rate 1e-4 \
  --weight_decay 1e-5 \
  --validation_size 0.15 \
  --latent_dim 1024 \
  --num_workers 0 \
  --verbose true \
  > run_logs/nifti_3d_autoencoder_nyu.log 2>&1 &
```

Watch progress:

```bash
tail -f run_logs/nifti_3d_autoencoder_nyu.log
```

## Full `dparsf` All-Sites Run on `soair`

```bash
cd /home/soair/XAI-for-ASD
source .venv/bin/activate
mkdir -p run_logs

nohup python3 -u nifti_3d_autoencoder.py \
  --manifest artifacts/nifti_manifest_dparsf.csv \
  --artifact_root artifacts/nifti_3d_autoencoder/all_sites_variance_latent1024 \
  --collapse_mode variance \
  --batch_size 2 \
  --epochs 100 \
  --patience 15 \
  --min_delta 1e-4 \
  --learning_rate 1e-4 \
  --weight_decay 1e-5 \
  --validation_size 0.15 \
  --latent_dim 1024 \
  --num_workers 0 \
  --verbose true \
  > run_logs/nifti_3d_autoencoder_all_sites.log 2>&1 &
```

## Quick Smoke Run

Use this before a long run if you want a fast end-to-end check:

```bash
python3 -u nifti_3d_autoencoder.py \
  --manifest artifacts/nifti_manifest_dparsf_nyu.csv \
  --artifact_root artifacts/nifti_3d_autoencoder/smoke \
  --batch_size 1 \
  --epochs 2 \
  --patience 1 \
  --latent_dim 64 \
  --max_subjects 8 \
  --verbose true
```

## Outputs

Each run writes:

- `summary.json`
- `training_history.csv`
- `best_autoencoder.pt`
- `latent_vectors.csv`

`latent_vectors.csv` contains one row per subject with metadata plus columns
named `latent_0000`, `latent_0001`, ..., suitable for downstream LOSO
classification experiments.
