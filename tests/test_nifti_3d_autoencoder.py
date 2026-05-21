import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

import nifti_3d_autoencoder


def create_synthetic_nifti(path: Path, seed: int, shape=(8, 8, 8, 6)) -> None:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    image = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(image, str(path))


class Nifti3dAutoencoderTests(unittest.TestCase):
    def build_manifest(self, root: Path, count: int = 4) -> Path:
        rows = []
        for index in range(count):
            site_id = "NYU" if index % 2 == 0 else "USM"
            label = index % 2
            nifti_path = root / f"subject_{index:03d}_func_preproc.nii.gz"
            create_synthetic_nifti(nifti_path, seed=100 + index)
            rows.append(
                {
                    "file_id": f"subject_{index:03d}",
                    "site_id": site_id,
                    "label": label,
                    "nifti_path": str(nifti_path),
                }
            )
        manifest_path = root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        return manifest_path

    def test_dataset_loads_variance_volume_and_normalizes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = self.build_manifest(temp_path, count=2)
            manifest = nifti_3d_autoencoder.load_manifest_dataframe(manifest_path)
            dataset = nifti_3d_autoencoder.NiftiManifestDataset(manifest)

            sample = dataset[0]
            volume = sample["volume"].numpy()

            self.assertEqual(volume.shape, (1, 8, 8, 8))
            self.assertAlmostEqual(float(volume.mean()), 0.0, places=5)
            self.assertAlmostEqual(float(volume.std()), 1.0, places=4)

    def test_model_returns_reconstruction_and_latent_with_expected_shapes(self):
        model = nifti_3d_autoencoder.NiftiAutoencoder3D(input_shape=(8, 8, 8), latent_dim=16)
        batch = torch.randn(2, 1, 8, 8, 8)

        reconstruction, latent = model(batch)

        self.assertEqual(reconstruction.shape, batch.shape)
        self.assertEqual(latent.shape, (2, 16))

    def test_run_autoencoder_experiment_writes_artifacts_and_latent_vectors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = self.build_manifest(temp_path, count=4)
            artifact_root = temp_path / "artifacts"
            config = nifti_3d_autoencoder.NiftiAutoencoderConfig(
                manifest=str(manifest_path),
                artifact_root=str(artifact_root),
                batch_size=2,
                epochs=1,
                patience=1,
                latent_dim=8,
                validation_size=0.25,
                num_workers=0,
                random_seed=123,
                verbose=False,
            )

            summary = nifti_3d_autoencoder.run_autoencoder_experiment(config)

            artifact_dir = Path(summary["artifact_dir"])
            self.assertTrue((artifact_dir / "summary.json").exists())
            self.assertTrue((artifact_dir / "training_history.csv").exists())
            self.assertTrue((artifact_dir / "best_autoencoder.pt").exists())
            self.assertTrue((artifact_dir / "latent_vectors.csv").exists())

            latent_frame = pd.read_csv(artifact_dir / "latent_vectors.csv")
            self.assertEqual(len(latent_frame), 4)
            latent_columns = [column for column in latent_frame.columns if column.startswith("latent_")]
            self.assertEqual(len(latent_columns), 8)


if __name__ == "__main__":
    unittest.main()
