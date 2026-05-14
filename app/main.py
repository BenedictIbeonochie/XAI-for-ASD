import argparse
import json
import random
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colormaps, colorbar
import collections
from itertools import product
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
import os
import pdb
import seaborn as sns
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients, DeepLiftShap, DeepLift, GradientShap, ShapleyValueSampling, ShapleyValues, FeatureAblation, GuidedBackprop, Occlusion
from nilearn import datasets, plotting
import networkx as nx
from functools import reduce
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity

try:
  from lime import lime_tabular
except ImportError:
  lime_tabular = None

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
DEFAULT_SEED = 2109459083
LEGACY_ARTIFACT_MESSAGE = (
  "Legacy globally selected feature artifacts are disabled in correction mode. "
  "Recompute fold-specific feature selections from the raw ROI time-series instead."
)
DEFAULT_INTERPRETATION_METHODS = (
  "Integrated Gradients",
  "SHAP",
  "LIME",
  "GuidedBackprop",
  "DeepLift",
  "DeepLiftShap",
  "GradientShap",
)

def get_data_from_abide(pipeline):
  downloads = f'abide/downloads/Outputs/{pipeline}/filt_global/rois_aal/'
  pheno_file = 'data/Phenotypic_V1_0b_preprocessed1.csv'

  if not os.path.isdir(downloads):
    raise FileNotFoundError(
      f"Could not find raw ROI time-series for pipeline '{pipeline}' at '{downloads}'. "
      "The corrected reanalysis requires the original subject-level ROI files rather than the legacy globally selected feature CSVs."
    )

  with open(pheno_file, 'r', encoding='utf-8') as phenotypic_file:
    pheno_list = phenotypic_file.readlines()

  labels_dict = {}
  for i in pheno_list[1:]:
    file_name = i.split(',')[6]
    diagnosis = i.split(',')[7]

    labels_dict[file_name] = float(diagnosis) # Save labels alongisde their filenames

  data = []
  labels = []

  for filename in sorted(os.listdir(downloads)):
    if filename.endswith('.1D'):  # Check if the file is a .1D file
      filepath = os.path.join(downloads, filename)
      dataset = np.loadtxt(filepath)  # Load the file
      data.append(dataset)  # Append the dataset to the list

      file_id = '_'.join(filename.split('_')[:-2]) # Get file ID from filename
      labels.append(labels_dict[file_id])

  labels = np.array(labels) - 1

  return data, labels
 
def get_feature_vecs(data):
  roi_size = data[0].shape[1]
  feature_vec_size = int(roi_size * (roi_size - 1) / 2)
  feature_vecs = []
  feature_indices = []

  vectorized_fisher_transfrom = np.vectorize(fishers_z_transform)
  
  for i in range(len(data)):
    corr_coefs = np.corrcoef(data[i], rowvar=False)
    corr_coefs = np.nan_to_num(corr_coefs)
    f = []
    idx = []

    transformed_corr_coefs = vectorized_fisher_transfrom(corr_coefs)

    lower_triangular_indices = np.tril_indices(transformed_corr_coefs.shape[0], -1)

    for row_idx, col_idx in zip(*lower_triangular_indices):  # Unpack indices
      coefficient = transformed_corr_coefs[row_idx, col_idx]
      f.append(coefficient)
      idx.append([row_idx, col_idx])

    feature_vecs.append(f)
    feature_indices.append(idx)

  feature_vecs = np.array(feature_vecs)
  feature_indices = np.array(feature_indices)

  return feature_vecs, feature_indices

def get_top_features_from_SVM_RFE(X, Y, indices, N, step, training_sample_indices=None):
  roi_lookup = prepare_feature_index_lookup(indices)
  svm = SVC(kernel="linear")
  rfe = RFE(estimator=svm, n_features_to_select=N, step=step, verbose=0)
  rfe.fit(X, Y)
  top_indices = np.where(rfe.support_)[0]
  top_rois = roi_lookup[top_indices].astype(int)

  if training_sample_indices is None:
    training_sample_indices = np.arange(len(X))

  return RFESelection(
    selected_feature_indices=np.asarray(top_indices, dtype=int),
    selected_roi_pairs=np.asarray(top_rois, dtype=int),
    training_sample_indices=np.asarray(training_sample_indices, dtype=int),
  )


def load_legacy_selected_features(*args, correction_mode=True, **kwargs):
  if correction_mode:
    raise RuntimeError(LEGACY_ARTIFACT_MESSAGE)
  raise NotImplementedError("Legacy artifact loading is not supported by the corrected workflow.")

def fishers_z_transform(x):
  # Handling the case where correlation coefficient is 1 or -1
  if x == 1:
    return np.inf
  elif x == -1:
    return -np.inf
  else:
    return 0.5 * np.log((1 + x) / (1 - x))
  
def safe_divide(numerator, denominator):
  """Safely divide two numbers, avoiding division by zero."""
  if denominator == 0:
      return 0
  else:
      return numerator / denominator


@dataclass
class ReanalysisConfig:
  n_splits: int = 5
  num_selected_features: int = 1000
  feature_count_candidates: tuple[int, ...] = ()
  feature_count_selection_metric: str = "f1"
  rfe_step: int = 20
  validation_size: float = 0.2
  batch_size: int = 128
  ae1_epochs: int = 50
  ae2_epochs: int = 50
  classifier_epochs: int = 300
  fine_tuning_epochs: int = 125
  early_stopping_patience: int = 15
  early_stopping_min_delta: float = 1e-4
  ae_learning_rate: float = 0.001
  classifier_learning_rate: float = 0.001
  fine_tuning_learning_rate: float = 0.0001
  weight_decay: float = 1e-4
  random_seed: int = DEFAULT_SEED
  ae1_hidden_size: int | None = 500
  ae2_hidden_size: int | None = 100
  use_feature_scaling: bool = True
  explanation_methods: tuple[str, ...] = DEFAULT_INTERPRETATION_METHODS
  explanation_top_n: int = 50
  explanation_background_samples: int = 100
  explanation_test_samples: int = 5
  artifact_root: str = "artifacts/reanalysis"
  save_artifacts: bool = True
  save_model_checkpoints: bool = False
  correction_mode: bool = True


@dataclass
class RFESelection:
  selected_feature_indices: np.ndarray
  selected_roi_pairs: np.ndarray
  training_sample_indices: np.ndarray


@dataclass
class ExplanationRanking:
  roi_pairs: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
  weights: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
  feature_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
  skipped_reason: str | None = None


@dataclass
class FoldResult:
  fold_id: int
  outer_train_indices: np.ndarray
  train_indices: np.ndarray
  validation_indices: np.ndarray
  test_indices: np.ndarray
  selection: RFESelection
  selected_feature_count: int
  metrics: dict[str, Any]
  true_labels: np.ndarray
  predicted_labels: np.ndarray
  train_feature_shape: tuple[int, int]
  validation_feature_shape: tuple[int, int]
  test_feature_shape: tuple[int, int]
  scaler_mean: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
  scaler_scale: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
  feature_count_tuning_records: list[dict[str, Any]] = field(default_factory=list)
  training_summary: dict[str, Any] = field(default_factory=dict)
  explanation_rankings: dict[str, ExplanationRanking] = field(default_factory=dict)
  artifact_dir: str | None = None
  model_checkpoint_path: str | None = None


@dataclass
class PipelineRunSummary:
  pipeline: str
  config: ReanalysisConfig
  fold_results: list[FoldResult]
  metrics_summary: dict[str, Any]
  interpretation_summary: dict[str, Any]
  artifact_dir: str | None = None
  legacy_artifacts_blocked: bool = True


def set_random_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if use_cuda:
    torch.cuda.manual_seed_all(seed)


def ensure_directory(path):
  path = Path(path)
  path.mkdir(parents=True, exist_ok=True)
  return path


def slugify_method_name(method_name):
  return method_name.lower().replace(" ", "_")


def dedupe_preserve_order(values):
  seen = set()
  ordered_values = []

  for value in values:
    if value in seen:
      continue
    seen.add(value)
    ordered_values.append(value)

  return tuple(ordered_values)


def normalize_sweep_candidates(candidates, default_value, cast_type):
  if candidates is None:
    return (cast_type(default_value),)

  normalized_values = []
  for candidate in candidates:
    value = cast_type(candidate)
    if isinstance(value, (int, float)) and value <= 0:
      continue
    normalized_values.append(value)

  if not normalized_values:
    normalized_values = [cast_type(default_value)]

  return dedupe_preserve_order(normalized_values)


def slugify_sweep_value(value):
  if isinstance(value, float):
    return format(value, '.6g').replace('.', 'p').replace('-', 'm')
  return str(value).replace('.', 'p').replace('-', 'm')


def prepare_feature_index_lookup(indices):
  indices = np.asarray(indices)
  if indices.ndim == 3:
    return indices[0]
  if indices.ndim == 2 and indices.shape[1] == 2:
    return indices
  raise ValueError("indices must have shape (samples, features, 2) or (features, 2)")


def coerce_explanation_ranking(roi_pairs, weights, feature_indices):
  roi_pairs = np.asarray(roi_pairs, dtype=int)
  if roi_pairs.size == 0:
    roi_pairs = np.empty((0, 2), dtype=int)
  elif roi_pairs.ndim == 1 and roi_pairs.shape[0] == 2:
    roi_pairs = roi_pairs.reshape(1, 2)
  elif roi_pairs.ndim != 2 or roi_pairs.shape[1] != 2:
    raise ValueError("roi_pairs must have shape (features, 2).")

  weights = np.asarray(weights, dtype=float).reshape(-1)
  feature_indices = np.asarray(feature_indices, dtype=int).reshape(-1)

  lengths = (len(roi_pairs), len(weights), len(feature_indices))
  aligned_length = min(lengths)

  if aligned_length == 0:
    if any(lengths):
      raise ValueError(
        "Explanation outputs must either all be empty or all contain at least one ranked feature."
      )
    return ExplanationRanking()

  roi_pairs = roi_pairs[:aligned_length]
  weights = weights[:aligned_length]
  feature_indices = feature_indices[:aligned_length]

  return ExplanationRanking(
    roi_pairs=roi_pairs,
    weights=weights,
    feature_indices=feature_indices,
  )


def sanitize_feature_matrix(features):
  return np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def apply_feature_selection(features, selection):
  return sanitize_feature_matrix(features)[:, selection.selected_feature_indices]


def get_candidate_feature_counts(config, max_feature_count):
  raw_candidates = config.feature_count_candidates if config.feature_count_candidates else (config.num_selected_features,)
  normalized_candidates = []

  for candidate in raw_candidates:
    candidate = int(candidate)
    if candidate <= 0:
      continue
    normalized_candidates.append(min(candidate, max_feature_count))

  if not normalized_candidates:
    normalized_candidates.append(min(config.num_selected_features, max_feature_count))

  return tuple(sorted(set(normalized_candidates)))


def scale_feature_sets(train_features, config, *other_feature_sets):
  train_features = sanitize_feature_matrix(train_features)
  other_feature_sets = [sanitize_feature_matrix(feature_set) for feature_set in other_feature_sets]

  if not config.use_feature_scaling:
    scaling_summary = {
      'enabled': False,
      'mean': np.empty(0, dtype=float),
      'scale': np.empty(0, dtype=float),
    }
    return train_features, other_feature_sets, scaling_summary

  scaler = StandardScaler()
  train_scaled = scaler.fit_transform(train_features)
  transformed_sets = [scaler.transform(feature_set) for feature_set in other_feature_sets]
  scaling_summary = {
    'enabled': True,
    'mean': np.asarray(scaler.mean_, dtype=float),
    'scale': np.asarray(scaler.scale_, dtype=float),
  }
  return train_scaled, transformed_sets, scaling_summary


def build_dataloader(features, labels, batch_size, shuffle):
  features_tensor = torch.tensor(np.asarray(features), dtype=torch.float32)
  labels_tensor = torch.tensor(np.asarray(labels), dtype=torch.long)
  dataset = TensorDataset(features_tensor, labels_tensor)
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def split_outer_train_validation(outer_train_indices, labels, validation_size, seed):
  outer_train_indices = np.asarray(outer_train_indices)
  stratify_labels = np.asarray(labels)[outer_train_indices]
  unique_labels, counts = np.unique(stratify_labels, return_counts=True)
  stratify = stratify_labels if len(unique_labels) > 1 and np.min(counts) > 1 else None

  train_indices, validation_indices = train_test_split(
    outer_train_indices,
    test_size=validation_size,
    random_state=seed,
    stratify=stratify,
  )

  return np.asarray(train_indices), np.asarray(validation_indices)


def resolve_hidden_layer_sizes(input_size, config):
  ae1_hidden_size = config.ae1_hidden_size if config.ae1_hidden_size is not None else min(500, input_size)
  ae1_hidden_size = max(1, min(ae1_hidden_size, max(1, input_size)))
  ae2_hidden_size = config.ae2_hidden_size if config.ae2_hidden_size is not None else min(100, ae1_hidden_size)
  ae2_hidden_size = max(1, min(ae2_hidden_size, ae1_hidden_size))
  return ae1_hidden_size, ae2_hidden_size


def compute_binary_metrics(true_labels, predicted_labels, positive_label=0):
  true_labels = np.asarray(true_labels).astype(int)
  predicted_labels = np.asarray(predicted_labels).astype(int)

  tp = int(np.sum((true_labels == positive_label) & (predicted_labels == positive_label)))
  tn = int(np.sum((true_labels != positive_label) & (predicted_labels != positive_label)))
  fp = int(np.sum((true_labels != positive_label) & (predicted_labels == positive_label)))
  fn = int(np.sum((true_labels == positive_label) & (predicted_labels != positive_label)))

  accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
  sensitivity = safe_divide(tp, tp + fn)
  specificity = safe_divide(tn, tn + fp)
  precision = safe_divide(tp, tp + fp)
  f1 = safe_divide((2 * precision * sensitivity), (precision + sensitivity))

  return {
    'accuracy': accuracy,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'precision': precision,
    'f1': f1,
    'confusion_matrix': np.array([[tp, fp], [fn, tn]]),
    'tp': tp,
    'fp': fp,
    'fn': fn,
    'tn': tn,
  }


def summarize_metrics(fold_results):
  metric_names = ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1')
  summary = {}

  for metric_name in metric_names:
    values = [fold_result.metrics[metric_name] for fold_result in fold_results]
    summary[metric_name] = {
      'mean': float(np.mean(values)),
      'std': float(np.std(values)),
      'values': [float(value) for value in values],
    }

  return summary


def to_serializable(value):
  if is_dataclass(value):
    return {key: to_serializable(val) for key, val in asdict(value).items()}
  if isinstance(value, pd.DataFrame):
    return value.to_dict(orient='records')
  if isinstance(value, dict):
    return {str(key): to_serializable(val) for key, val in value.items()}
  if isinstance(value, (list, tuple)):
    return [to_serializable(item) for item in value]
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (np.integer,)):
    return int(value)
  if isinstance(value, (np.floating,)):
    return float(value)
  if isinstance(value, Path):
    return str(value)
  return value


def write_json_file(path, payload):
  path = Path(path)
  ensure_directory(path.parent)
  with open(path, 'w', encoding='utf-8') as json_file:
    json.dump(to_serializable(payload), json_file, indent=2)
  

def clone_module_state(module):
  return {
    key: value.detach().cpu().clone()
    for key, value in module.state_dict().items()
  }


def compute_average_loss(model, dataloader, criterion):
  model.eval()
  total_loss = 0.0
  total_samples = 0

  with torch.no_grad():
    for data, labels in dataloader:
      data = data.float().to(device)
      labels = labels.long().to(device)
      outputs = model(data)
      loss = criterion(outputs, labels)
      batch_size = labels.shape[0]
      total_loss += float(loss.item()) * batch_size
      total_samples += batch_size

  if total_samples == 0:
    return float('inf')

  return total_loss / total_samples


def train_supervised_stage(model, train_dataloader, val_dataloader, criterion, optimizer, max_epochs, patience, min_delta, stage_name, verbose=False):
  best_state = clone_module_state(model)
  best_val_loss = float('inf')
  best_epoch = 0
  epochs_without_improvement = 0
  history = []

  for epoch in range(max_epochs):
    model.train()
    total_train_loss = 0.0
    total_train_samples = 0

    for data, labels in train_dataloader:
      data = data.float().to(device)
      labels = labels.long().to(device)
      optimizer.zero_grad()
      outputs = model(data)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      batch_size = labels.shape[0]
      total_train_loss += float(loss.item()) * batch_size
      total_train_samples += batch_size

    average_train_loss = safe_divide(total_train_loss, total_train_samples)
    val_loss = compute_average_loss(model, val_dataloader, criterion)
    improved = val_loss < (best_val_loss - min_delta)

    history.append({
      'epoch': epoch + 1,
      'train_loss': float(average_train_loss),
      'validation_loss': float(val_loss),
      'improved': bool(improved),
    })

    if improved:
      best_val_loss = val_loss
      best_epoch = epoch + 1
      best_state = clone_module_state(model)
      epochs_without_improvement = 0
    else:
      epochs_without_improvement += 1

    if verbose:
      print(
        f"{stage_name} epoch {epoch + 1}/{max_epochs} "
        f"train_loss={average_train_loss:.6f} val_loss={val_loss:.6f}"
      )

    if patience and epochs_without_improvement >= patience:
      break

  model.load_state_dict(best_state)

  return {
    'stage_name': stage_name,
    'epochs_trained': len(history),
    'best_epoch': best_epoch,
    'best_validation_loss': float(best_val_loss),
    'history': history,
  }


class Autoencoder(nn.Module):
  def __init__(self, input_size, encoded_output_size, rho=0.2, beta=2, criterion=nn.MSELoss()):
    """
    rho: desired sparsity parameter
    beta: weight of the KL divergence term
    """
    super(Autoencoder, self).__init__()

    self.encoder = nn.Linear(input_size, encoded_output_size)
    self.decoder = nn.Linear(encoded_output_size, input_size)
    self.rho = rho
    self.beta = beta
    self.criterion = criterion
  
  def kl_divergence(self, rho, rho_hat):
    """Calculates KL divergence for regularization."""
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)) 
  
  def forward(self, x):
    encoded = torch.relu(self.encoder(x))

    # Compute average activation of hidden neurons
    rho_hat = torch.mean(encoded, dim=0) 

    kl_loss = self.kl_divergence(self.rho, rho_hat).sum()

    decoded = self.decoder(encoded)

    # Total loss: Reconstruction loss + KL divergence
    mse_loss = self.criterion(decoded, x)
    loss = mse_loss + self.beta * kl_loss 

    return encoded, decoded, loss

class SoftmaxClassifier(nn.Module):
  def __init__(self, input_size, num_classes):
    super(SoftmaxClassifier, self).__init__()
    self.linear = nn.Linear(input_size, num_classes)

  def forward(self, x):
    out = self.linear(x)
    return out
  
class StackedAutoencoder(nn.Module):
  def __init__(self, AE1, AE2, classifier):
      super(StackedAutoencoder, self).__init__()
      self.ae1 = AE1  # Assuming you have your pre-trained AE1
      self.ae2 = AE2  # Assuming you have your pre-trained AE2
      self.classifier = classifier 

  def forward(self, x):
      x = torch.relu(self.ae1.encoder(x))  # Match the non-linearity used during autoencoder pretraining
      x = torch.relu(self.ae2.encoder(x))  # Preserve the learned staged representation
      x = self.classifier(x)
      return x
  
class CustomDataset(Dataset):
  def __init__(self, data, labels):
    'Initialization'
    self.labels = labels
    self.data = data

  def __len__(self):
    return len(self.data.indices)
  
  def __getitem__(self, idx):
    data_idx = self.data.indices[idx]  # Get index into the original dataset
    labels_idx = self.labels.indices[idx]  # Get index into the original dataset
    return self.data.dataset[data_idx], self.labels.dataset[labels_idx] 


def encode_data(dataloader, ae1, ae2, device):
  encoded_data = []
  labels = []

  for batch in dataloader:
    data, label = batch
    data = data.float().to(device)

    with torch.no_grad():
      encoded_features, _, __ = ae1(data)
      encoded_features, _, __ = ae2(encoded_features)
      encoded_data.append(encoded_features)
      labels.append(label)

  return encoded_data, labels

def get_encoded_data(model, dataloader, dataloader_params, device):
  """Encodes data from a dataloader using a given model.
  
  Args:
      model: The PyTorch model used for encoding.
      dataloader: The PyTorch dataloader containing the data.
      device: The device (e.g., 'cuda:0' or 'cpu') where the model and data should be sent.

  Returns:
      A tuple: (encoded_dataset, encoded_dataset_loader), where
          * encoded_dataset is a TensorDataset containing the encoded features and labels.
          * encoded_dataset_loader is a DataLoader for the encoded_dataset.
  """  

  encoded_features_from_model = []
  labels_from_model = []

  for batch in dataloader:
    data, labels = batch
    data = data.float().to(device) 

    with torch.no_grad():
      encoded_features, _, __ = model(data)  # Assuming your model outputs encoded features, ...
      encoded_features_from_model.append(encoded_features)
      labels_from_model.append(labels)

  encoded_dataset_tensor = torch.cat(encoded_features_from_model, dim=0)
  labels_tensor = torch.cat(labels_from_model, dim=0)

  encoded_dataset = TensorDataset(encoded_dataset_tensor, labels_tensor)
  encoded_dataset_loader = DataLoader(encoded_dataset, **dataloader_params)

  return encoded_dataset, encoded_dataset_loader

def find_top_rois_using_LIME(N, model, test_dataloader, train_dataloader, rois):
  if lime_tabular is None:
    raise ImportError("LIME is not installed in this environment.")
  features_list = []
  labels_list = []

  for features, labels in train_dataloader:
      # Move data to CPU if it's on a GPU
      features = features.cpu().numpy()
      labels = labels.cpu().numpy()
      
      # Append to lists
      features_list.append(features)
      labels_list.append(labels)

  # Concatenate all features and labels
  X_train = np.concatenate(features_list, axis=0)
  Y_train = np.concatenate(labels_list, axis=0)

  # Assuming 'dataloader' is your DataLoader instance
  features_list = []
  labels_list = []

  for features, labels in test_dataloader:
      # Move data to CPU if it's on a GPU
      features = features.cpu().numpy()
      labels = labels.cpu().numpy()
      
      # Append to lists
      features_list.append(features)
      labels_list.append(labels)

  # Concatenate all features and labels
  X_test = np.concatenate(features_list, axis=0)
  Y_test = np.concatenate(labels_list, axis=0)

  # Initialize LIME Explainer for tabular data
  explainer = lime_tabular.LimeTabularExplainer(
      training_data=X_train,  # Use your training data here
      feature_names=list(range(X_train.shape[1])),  # Feature names or indices
      class_names=['Class 0', 'Class 1'],  # Output classes
      mode='classification'
  )

  # Select an instance to explain
  instance_index = 0  # Example index, choose appropriately
  instance = X_test[instance_index]

  # Generate explanation for the selected instance
  explanation = explainer.explain_instance(
      data_row=instance, 
      predict_fn=lambda batch: model_predict_lime(model, batch),
      num_features=min(1000, X_train.shape[1]),  # Number of top features you want to show
      top_labels=1  # Number of top labels for multi-class classification
  )

  feature_weights = explanation.as_list(label=explanation.top_labels[0])

  sorted_features = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)

  sorted_feature_indices = [find_index_from_string(feature[0]) for feature in sorted_features]

  sorted_feature_weights = [abs(feature[1]) for feature in sorted_features]

  return rois[sorted_feature_indices[:N]], sorted_feature_weights[:N], sorted_feature_indices[:N]

def find_index_from_string(stri):
  stri = stri.split(' ')

  for i in stri:
    if i.isnumeric() and (float(i) >= 1 or float(i) == 0):
      return int(i)

def find_top_rois_using_SHAP(N, model, test_dataloader, train_dataloader, rois):
  # Select a background dataset from train_dataloader
  background_data = []
  for batch in train_dataloader:
      data, labels = batch
      data = data.float().to(device) 
      labels = labels.long().to(device)

      background_data.append(data)
      if len(background_data) >= 100:  # Collect 100 samples, adjust as needed
          break
  background_data = torch.cat(background_data)[:100]  # Adjust size as needed

  # Select test instances from test_dataloader
  test_instances = []
  for batch in test_dataloader:
      data, labels = batch
      data = data.float().to(device) 
      labels = labels.long().to(device) 

      test_instances.append(data)
      if len(test_instances) >= 5:  # Let's say we want to explain 5 test instances
          break
  test_instances = torch.cat(test_instances)[:5]  # Adjust size as needed

  background_data = background_data.to(device)
  test_instances = test_instances.to(device)

  # Initialize SHAP DeepExplainer
  explainer = shap.DeepExplainer(model, background_data)

  # Compute SHAP values for test_instances
  shap_values = explainer.shap_values(test_instances)

  mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

  feature_importance = np.mean(mean_abs_shap_values, axis=1)

  # Step 4: Find indices of top 100 features
  top_indices = np.argsort(feature_importance)[-(N):][::-1]

  return rois[top_indices], feature_importance[top_indices], top_indices
  
def find_top_rois_using_integrated_gradients(N, model, test_dataloader, rois):
  for batch in test_dataloader:
    data, labels = batch
    data = data.float().to(device) 
    labels = labels.long().to(device) 
    break

  model.eval()
  data.requires_grad = True  # Enable gradient computation on the input

  # Baseline (here, a tensor of zeros)
  baseline = torch.zeros_like(data).to(device)

  # Initialize Integrated Gradients with model
  integrated_gradients = IntegratedGradients(model)

  # Compute attributions for the autism positive class
  # Assuming the first output (index 0) corresponds to autism positive
  # Convergence delta is approximate error
  attributions_ig, delta = integrated_gradients.attribute(data, baselines=baseline, target=0, return_convergence_delta=True)

  # Calculate the mean of the attributions across all input samples to get an average importance
  # Postive attribution means positive contribution of that feature
  # Negative attribution means negative contribution of that feature
  # 0 attribution means 0 contribution of that feature
  attributions_mean = attributions_ig.mean(dim=0).cpu().detach().numpy()

  # To get top features we need to take the features in descending order of attribution value to get most contributing features
  abs_attribution = np.abs(attributions_mean)

  top_indices = np.argsort(abs_attribution)[-(N):][::-1]

  return rois[top_indices], abs_attribution[top_indices], top_indices

def find_top_rois_using_DeepLift(N, model, test_dataloader, rois):
  for batch in test_dataloader:
    data, labels = batch
    data = data.float().to(device) 
    labels = labels.long().to(device) 
    break

  model.eval()
  data.requires_grad = True  # Enable gradient computation on the input

  deep_lift = DeepLift(model)
  attributions_dl = deep_lift.attribute(data, target=0)

  attributions_mean = attributions_dl.mean(dim=0).cpu().detach().numpy()

  abs_attribution = np.abs(attributions_mean)

  top_indices = np.argsort(abs_attribution)[-(N):][::-1]

  return rois[top_indices], abs_attribution[top_indices], top_indices

def find_top_rois_using_DeepLiftShap(N, model, test_dataloader, rois):
  for batch in test_dataloader:
    data, labels = batch
    data = data.float().to(device) 
    labels = labels.long().to(device) 
    break

  model.eval()
  data.requires_grad = True  # Enable gradient computation on the input

  baseline = torch.zeros_like(data).to(device)

  deep_lift_shap = DeepLiftShap(model)
  attributions_dls = deep_lift_shap.attribute(data, baselines=baseline, target=0)

  attributions_mean = attributions_dls.mean(dim=0).cpu().detach().numpy()

  abs_attribution = np.abs(attributions_mean)

  top_indices = np.argsort(abs_attribution)[-(N):][::-1]

  return rois[top_indices], abs_attribution[top_indices], top_indices

def find_top_rois_using_GradientShap(N, model, test_dataloader, rois):
  for batch in test_dataloader:
    data, labels = batch
    data = data.float().to(device) 
    labels = labels.long().to(device) 
    break

  model.eval()
  data.requires_grad = True  # Enable gradient computation on the input

  baseline = torch.zeros_like(data).to(device)

  gradient_shap = GradientShap(model)

  attributions_gs = gradient_shap.attribute(data, baselines=baseline, target=0)

  attributions_mean = attributions_gs.mean(dim=0).cpu().detach().numpy()

  abs_attribution = np.abs(attributions_mean)

  top_indices = np.argsort(abs_attribution)[-(N):][::-1]

  return rois[top_indices], abs_attribution[top_indices], top_indices

def find_top_rois_using_GuidedBackprop(N, model, test_dataloader, rois):
  for batch in test_dataloader:
    data, labels = batch
    data = data.float().to(device) 
    labels = labels.long().to(device) 
    break

  model.eval()
  data.requires_grad = True  # Enable gradient computation on the input

  baseline = torch.zeros_like(data).to(device)

  guided_backprop = GuidedBackprop(model)

  attributions_gb = guided_backprop.attribute(data, target=0)

  attributions_mean = attributions_gb.mean(dim=0).cpu().detach().numpy()

  abs_attribution = np.abs(attributions_mean)

  top_indices = np.argsort(abs_attribution)[-(N):][::-1]

  return rois[top_indices], abs_attribution[top_indices], top_indices


def get_interpretability_method_map():
  return {
    "Integrated Gradients": lambda top_n, model, test_dataloader, train_dataloader, rois: find_top_rois_using_integrated_gradients(top_n, model, test_dataloader, rois),
    "SHAP": lambda top_n, model, test_dataloader, train_dataloader, rois: find_top_rois_using_SHAP(top_n, model, test_dataloader, train_dataloader, rois),
    "LIME": lambda top_n, model, test_dataloader, train_dataloader, rois: find_top_rois_using_LIME(top_n, model, test_dataloader, train_dataloader, rois),
    "GuidedBackprop": lambda top_n, model, test_dataloader, train_dataloader, rois: find_top_rois_using_GuidedBackprop(top_n, model, test_dataloader, rois),
    "DeepLift": lambda top_n, model, test_dataloader, train_dataloader, rois: find_top_rois_using_DeepLift(top_n, model, test_dataloader, rois),
    "DeepLiftShap": lambda top_n, model, test_dataloader, train_dataloader, rois: find_top_rois_using_DeepLiftShap(top_n, model, test_dataloader, rois),
    "GradientShap": lambda top_n, model, test_dataloader, train_dataloader, rois: find_top_rois_using_GradientShap(top_n, model, test_dataloader, rois),
  }

def get_threshold_from_percentile(adjacency_matrix, percentile):
  all_weights = adjacency_matrix[np.nonzero(adjacency_matrix)]
  threshold = np.percentile(all_weights, percentile) 
  return threshold

def expand_relative_coords(coordinates, percent):
  # Calculate center
  center = np.mean(coordinates, axis=0)

  # Center the coordinates
  centered_coordinates = coordinates - center 

  # Scale the coordinates
  scaled_coordinates = centered_coordinates * percent

  # Shift back to original center
  spread_coordinates = scaled_coordinates + center 

  return spread_coordinates


def print_connections(rois, weights, method, pipeline, top_regions=50, top_regions_df=10, show_now=False, save=False, print_graph=True):
  atlas = datasets.fetch_atlas_aal(version='SPM12')
  labels = atlas.labels  # List of AAL region labels
  weights = np.array(weights)
  rois = rois.astype(int)[:top_regions]

  weights = ((weights - weights.min()) / (weights.max() - weights.min())) * 10

  weights = weights[:top_regions]

  edge_cmap = colormaps['viridis']  # Colormap choice 

  num_connections = len(rois)

  cmap = colormaps['viridis']

  if print_graph:
    fig = plt.figure(figsize=(15, 8))
    ax_connection_connectome = fig.add_axes([0.05, 0.55, 0.8, 0.40])
    ax_connection_colorbar = fig.add_axes([0.85, 0.55, 0.05, 0.40])

    ax_roi_connectome = fig.add_axes([0.05, 0.05, 0.8, 0.40])
    ax_roi_colorbar = fig.add_axes([0.85, 0.05, 0.05, 0.40])

    # Set the figure-wide title
    fig.suptitle(f'Top {num_connections} connections and ROI Importance using {method} for {pipeline}', fontsize=16)


  G = nx.Graph()

  # Add nodes (brain regions)
  for label in labels:
    G.add_node(label)

  # Add edges with weights
  for i, roi_pair in enumerate(rois):
      roi1_index = int(roi_pair[0])
      roi2_index = int(roi_pair[1])
      roi1_name = labels[roi1_index]
      roi2_name = labels[roi2_index]
      weight = weights[i]
      G.add_edge(roi1_name, roi2_name, weight=weight) 

  node_color = 'grey'

  coordinates = expand_relative_coords(plotting.find_parcellation_cut_coords(atlas.maps), 1.08) 

  adjacency_matrix = nx.adjacency_matrix(G).todense()

  # Dynamic Thresholding
  edge_threshold = get_threshold_from_percentile(adjacency_matrix, 0)  # Show all 

  if print_graph:
    plotting.plot_connectome(adjacency_matrix, coordinates,
                            node_color=node_color,
                            edge_vmin=0,
                            edge_vmax=weights.max(),
                            edge_cmap=edge_cmap,
                            edge_threshold=edge_threshold,
                            axes=ax_connection_connectome)
  
    norm = Normalize(vmin=weights.min(), vmax=weights.max())

    cb = colorbar.ColorbarBase(ax_connection_colorbar, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb.set_label('Importance')

  # Count the occurrence of each ROI
  roi_counts = np.zeros(len(labels))
  all_rois = [int(roi) for pair in rois for roi in pair]  # Flatten list of ROI pairs
  roi_counts = collections.Counter(all_rois)  # Count occurrences of each ROI 
  top_rois, top_counts = zip(*roi_counts.most_common())

  adjacency_matrix = np.zeros((len(coordinates), len(coordinates)))  # No edges

  roi_importances = []

  for idx, label in enumerate(labels):
    if idx in top_rois:
      i = top_rois.index(idx)

      weight = 1

      possible_weights = []
      count_in_rois = 0

      for j in range(len(rois)):
        if float(idx) in rois[j]:
          count_in_rois += 1  
          possible_weights.append(weights[j])  

      if count_in_rois >= 1:
        weight = max(possible_weights)

      roi_importances.append((top_counts[i]+1)*weight)      
    else:
      roi_importances.append(0.0001)
  
  roi_importances = np.array(roi_importances)

  # Normalize the importance scores for node sizes and colors
  normalized_sizes = 20 + (roi_importances - roi_importances.min()) / (roi_importances.max() - roi_importances.min()) * 180  # Scale between 20 and 200

  normalized_colors = cmap((roi_importances - roi_importances.min()) / (roi_importances.max() - roi_importances.min()))

  if print_graph:
    plotting.plot_connectome(adjacency_matrix, coordinates,
                          node_color=normalized_colors,
                          node_size=normalized_sizes,
                          display_mode='ortho',
                          colorbar=False,
                          axes=ax_roi_connectome)

    norm = Normalize(vmin=roi_importances.min(), vmax=roi_importances.max())

    cb = colorbar.ColorbarBase(ax_roi_colorbar, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb.set_label('Importance')

    if save:
      filename = f"plots/{pipeline}/{pipeline}_plot_{method}_{num_connections}_connections.png"
      plt.savefig(filename)

    if show_now:
      plt.show()

  # Open the JSON file for reading
  with open('aal_roi_functions.json', 'r') as file:
    # Parse the JSON file into a Python dictionary
    ROI_functions = json.load(file)

  labels = np.array(atlas.labels)
  top_connections = labels[rois[:top_regions_df]]
  connections_with_weights = np.array([(connection[0], ROI_functions[connection[0]], connection[1], ROI_functions[connection[1]], np.round(weight, 2)) for connection, weight in zip(top_connections, weights)])

  # Convert the top connections to a DataFrame for nice formatting
  top_connections_df = pd.DataFrame(connections_with_weights, columns=['ROI 1', 'ROI 1 function', 'ROI 2', 'ROI 2 function', 'Importance'])

  important_rois = np.argsort(roi_importances)[::-1]
  important_rois_weights = roi_importances[important_rois]
  
  important_rois_weights = 10 * ((important_rois_weights - important_rois_weights.min()) / (important_rois_weights.max() - important_rois_weights.min()))

  important_rois_with_weights = np.array([(labels[roi], ROI_functions[labels[roi]], np.round(weight, 2)) for roi, weight in zip(important_rois, important_rois_weights)])

  top_rois_df = pd.DataFrame(important_rois_with_weights[:top_regions_df], columns=['ROI', 'Function', 'Importance'])

  return top_connections_df, top_rois_df

def model_predict_lime(model, data):
  # Convert data to tensor, pass through model, and return softmax probabilities
  data_tensor = torch.tensor(data).float().to(device)
  model.eval()
  with torch.no_grad():
    outputs = model(data_tensor)
    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

  return probabilities

def train_single_fold_model(train_dataloader, val_dataloader, input_size, config, verbose=False):
  ae1_hidden_size, ae2_hidden_size = resolve_hidden_layer_sizes(input_size, config)

  ae1 = Autoencoder(input_size, ae1_hidden_size).to(device)
  ae2 = Autoencoder(ae1_hidden_size, ae2_hidden_size).to(device)
  classifier = SoftmaxClassifier(ae2_hidden_size, 2).to(device)
  model = StackedAutoencoder(ae1, ae2, classifier).to(device)

  optimizer_ae1 = optim.Adam(ae1.parameters(), lr=config.ae_learning_rate, weight_decay=config.weight_decay)
  optimizer_ae2 = optim.Adam(ae2.parameters(), lr=config.ae_learning_rate, weight_decay=config.weight_decay)
  optimizer_classifier = optim.Adam(classifier.parameters(), lr=config.classifier_learning_rate, weight_decay=config.weight_decay)
  optimizer_model = optim.Adam(model.parameters(), lr=config.fine_tuning_learning_rate, weight_decay=config.weight_decay)
  classifier_criterion = nn.CrossEntropyLoss()
  training_summary = {
    'ae1': {'epochs_trained': config.ae1_epochs},
    'ae2': {'epochs_trained': config.ae2_epochs},
  }

  for epoch in range(config.ae1_epochs):
    for data, _ in train_dataloader:
      data = data.float().to(device)
      optimizer_ae1.zero_grad()
      _, _, loss = ae1(data)
      loss.backward()
      optimizer_ae1.step()

    if verbose:
      print(f"AE1 epoch {epoch + 1}/{config.ae1_epochs} loss={loss.item():.6f}")

  encoded_train_dataset, encoded_train_loader = get_encoded_data(
    ae1,
    train_dataloader,
    {'batch_size': config.batch_size, 'shuffle': True, 'num_workers': 0},
    device,
  )
  _, encoded_val_loader = get_encoded_data(
    ae1,
    val_dataloader,
    {'batch_size': config.batch_size, 'shuffle': False, 'num_workers': 0},
    device,
  )

  for epoch in range(config.ae2_epochs):
    for data, _ in encoded_train_loader:
      data = data.float().to(device)
      optimizer_ae2.zero_grad()
      _, _, loss = ae2(data)
      loss.backward()
      optimizer_ae2.step()

    if verbose:
      print(f"AE2 epoch {epoch + 1}/{config.ae2_epochs} loss={loss.item():.6f}")

  _, encoded_classifier_train_loader = get_encoded_data(
    ae2,
    encoded_train_loader,
    {'batch_size': config.batch_size, 'shuffle': True, 'num_workers': 0},
    device,
  )
  _, encoded_classifier_val_loader = get_encoded_data(
    ae2,
    encoded_val_loader,
    {'batch_size': config.batch_size, 'shuffle': False, 'num_workers': 0},
    device,
  )

  training_summary['classifier'] = train_supervised_stage(
    classifier,
    encoded_classifier_train_loader,
    encoded_classifier_val_loader,
    classifier_criterion,
    optimizer_classifier,
    config.classifier_epochs,
    config.early_stopping_patience,
    config.early_stopping_min_delta,
    "Classifier",
    verbose=verbose,
  )

  training_summary['fine_tuning'] = train_supervised_stage(
    model,
    train_dataloader,
    val_dataloader,
    classifier_criterion,
    optimizer_model,
    config.fine_tuning_epochs,
    config.early_stopping_patience,
    config.early_stopping_min_delta,
    "Fine-tuning",
    verbose=verbose,
  )

  return model, training_summary


def evaluate_model(model, test_dataloader):
  model.eval()
  true_labels = []
  predicted_labels = []

  with torch.no_grad():
    for data, labels in test_dataloader:
      data = data.float().to(device)
      labels = labels.long().to(device)
      outputs = model(data)
      _, predicted = torch.max(outputs.data, 1)
      true_labels.extend(labels.cpu().numpy())
      predicted_labels.extend(predicted.cpu().numpy())

  true_labels = np.asarray(true_labels, dtype=int)
  predicted_labels = np.asarray(predicted_labels, dtype=int)
  metrics = compute_binary_metrics(true_labels, predicted_labels)

  return metrics, true_labels, predicted_labels


def select_feature_count_for_fold(feature_vectors, labels_from_abide, roi_lookup, train_indices, validation_indices, config, verbose=False):
  candidate_feature_counts = get_candidate_feature_counts(config, feature_vectors.shape[1])

  if len(candidate_feature_counts) == 1:
    return candidate_feature_counts[0], []

  tuning_records = []
  valid_metric_names = {'accuracy', 'sensitivity', 'specificity', 'precision', 'f1'}
  metric_name = config.feature_count_selection_metric if config.feature_count_selection_metric in valid_metric_names else 'f1'
  best_metric_value = float('-inf')
  best_accuracy = float('-inf')
  best_feature_count = candidate_feature_counts[0]

  for feature_count in candidate_feature_counts:
    candidate_selection = get_top_features_from_SVM_RFE(
      feature_vectors[train_indices],
      labels_from_abide[train_indices],
      roi_lookup,
      feature_count,
      config.rfe_step,
      training_sample_indices=train_indices,
    )

    candidate_train_features = apply_feature_selection(feature_vectors[train_indices], candidate_selection)
    candidate_validation_features = apply_feature_selection(feature_vectors[validation_indices], candidate_selection)
    candidate_train_features, [candidate_validation_features], _ = scale_feature_sets(
      candidate_train_features,
      config,
      candidate_validation_features,
    )

    candidate_train_labels = labels_from_abide[train_indices]
    candidate_validation_labels = labels_from_abide[validation_indices]
    candidate_train_loader = build_dataloader(candidate_train_features, candidate_train_labels, config.batch_size, shuffle=True)
    candidate_validation_loader = build_dataloader(candidate_validation_features, candidate_validation_labels, config.batch_size, shuffle=False)

    candidate_model, candidate_training_summary = train_single_fold_model(
      candidate_train_loader,
      candidate_validation_loader,
      input_size=candidate_train_features.shape[1],
      config=config,
      verbose=False,
    )
    candidate_metrics, _, _ = evaluate_model(candidate_model, candidate_validation_loader)
    metric_value = float(candidate_metrics[metric_name])
    accuracy_value = float(candidate_metrics['accuracy'])
    tuning_records.append({
      'feature_count': int(feature_count),
      'selection_training_sample_indices': np.asarray(train_indices, dtype=int),
      'validation_metrics': candidate_metrics,
      'training_summary': candidate_training_summary,
    })

    if verbose:
      print(
        f"Feature-count candidate {feature_count}: "
        f"validation_{metric_name}={metric_value:.4f}, "
        f"validation_accuracy={accuracy_value:.4f}"
      )

    if (
      metric_value > best_metric_value
      or (
        np.isclose(metric_value, best_metric_value)
        and (
          accuracy_value > best_accuracy
          or (
            np.isclose(accuracy_value, best_accuracy)
            and feature_count < best_feature_count
          )
        )
      )
    ):
      best_metric_value = metric_value
      best_accuracy = accuracy_value
      best_feature_count = int(feature_count)

  return best_feature_count, tuning_records


def compute_fold_explanations(model, train_dataloader, test_dataloader, selected_roi_pairs, config):
  explanation_rankings = {}
  selected_roi_pairs = np.asarray(selected_roi_pairs, dtype=int)
  method_map = get_interpretability_method_map()

  for method_name in config.explanation_methods:
    if method_name not in method_map:
      explanation_rankings[method_name] = ExplanationRanking(
        skipped_reason=f"Unknown interpretation method '{method_name}'.",
      )
      continue

    try:
      roi_pairs, weights, feature_indices = method_map[method_name](
        config.explanation_top_n,
        model,
        test_dataloader,
        train_dataloader,
        selected_roi_pairs,
      )
      explanation_rankings[method_name] = coerce_explanation_ranking(
        roi_pairs,
        weights,
        feature_indices,
      )
    except Exception as error:
      explanation_rankings[method_name] = ExplanationRanking(
        skipped_reason=str(error),
      )

  return explanation_rankings


def aggregate_explanations_across_folds(fold_results, top_n):
  aggregated_results = {}
  method_names = sorted({
    method_name
    for fold_result in fold_results
    for method_name in fold_result.explanation_rankings.keys()
  })

  for method_name in method_names:
    connection_records = []
    roi_records = []
    skipped_folds = []

    for fold_result in fold_results:
      explanation = fold_result.explanation_rankings.get(method_name)
      if explanation is None:
        continue
      if explanation.skipped_reason:
        skipped_folds.append({'fold_id': fold_result.fold_id, 'reason': explanation.skipped_reason})
        continue

      for rank, (roi_pair, weight) in enumerate(zip(explanation.roi_pairs[:top_n], explanation.weights[:top_n]), start=1):
        roi_1, roi_2 = sorted(int(value) for value in roi_pair)
        connection_records.append({
          'fold_id': fold_result.fold_id,
          'rank': rank,
          'roi_1_index': roi_1,
          'roi_2_index': roi_2,
          'importance': float(weight),
        })
        roi_records.append({
          'fold_id': fold_result.fold_id,
          'rank': rank,
          'roi_index': roi_1,
          'importance': float(weight),
        })
        roi_records.append({
          'fold_id': fold_result.fold_id,
          'rank': rank,
          'roi_index': roi_2,
          'importance': float(weight),
        })

    if connection_records:
      connections_df = pd.DataFrame(connection_records)
      aggregated_connections = (
        connections_df
        .groupby(['roi_1_index', 'roi_2_index'], as_index=False)
        .agg(
          fold_occurrence=('fold_id', 'nunique'),
          mean_importance=('importance', 'mean'),
          median_rank=('rank', 'median'),
        )
        .sort_values(['fold_occurrence', 'mean_importance', 'median_rank'], ascending=[False, False, True])
        .reset_index(drop=True)
      )
    else:
      aggregated_connections = pd.DataFrame(columns=['roi_1_index', 'roi_2_index', 'fold_occurrence', 'mean_importance', 'median_rank'])

    if roi_records:
      rois_df = pd.DataFrame(roi_records)
      aggregated_rois = (
        rois_df
        .groupby('roi_index', as_index=False)
        .agg(
          fold_occurrence=('fold_id', 'nunique'),
          mean_importance=('importance', 'mean'),
          median_rank=('rank', 'median'),
        )
        .sort_values(['fold_occurrence', 'mean_importance', 'median_rank'], ascending=[False, False, True])
        .reset_index(drop=True)
      )
    else:
      aggregated_rois = pd.DataFrame(columns=['roi_index', 'fold_occurrence', 'mean_importance', 'median_rank'])

    aggregated_results[method_name] = {
      'connections': aggregated_connections,
      'rois': aggregated_rois,
      'folds_aggregated': int(len(fold_results) - len(skipped_folds)),
      'skipped_folds': skipped_folds,
    }

  return aggregated_results


def write_fold_artifacts(fold_result, artifact_dir):
  artifact_dir = ensure_directory(artifact_dir)
  np.savetxt(artifact_dir / 'selected_feature_indices.csv', fold_result.selection.selected_feature_indices, delimiter=',', fmt='%d')
  np.savetxt(artifact_dir / 'selected_roi_pairs.csv', fold_result.selection.selected_roi_pairs, delimiter=',', fmt='%d')
  if len(fold_result.scaler_mean):
    np.savetxt(artifact_dir / 'scaler_mean.csv', fold_result.scaler_mean, delimiter=',')
    np.savetxt(artifact_dir / 'scaler_scale.csv', fold_result.scaler_scale, delimiter=',')

  prediction_df = pd.DataFrame({
    'sample_index': fold_result.test_indices,
    'true_label': fold_result.true_labels,
    'predicted_label': fold_result.predicted_labels,
  })
  prediction_df.to_csv(artifact_dir / 'predictions.csv', index=False)
  write_json_file(artifact_dir / 'metrics.json', fold_result.metrics)
  write_json_file(
    artifact_dir / 'feature_count_tuning.json',
    {
      'selected_feature_count': fold_result.selected_feature_count,
      'candidates': fold_result.feature_count_tuning_records,
    },
  )
  write_json_file(artifact_dir / 'training_summary.json', fold_result.training_summary)

  for method_name, explanation in fold_result.explanation_rankings.items():
    method_slug = slugify_method_name(method_name)
    if explanation.skipped_reason:
      write_json_file(
        artifact_dir / f'{method_slug}_explanation.json',
        {'skipped_reason': explanation.skipped_reason},
      )
      continue

    explanation_df = pd.DataFrame({
      'feature_rank': np.arange(1, len(explanation.weights) + 1),
      'feature_index': explanation.feature_indices,
      'roi_1_index': explanation.roi_pairs[:, 0] if len(explanation.roi_pairs) else [],
      'roi_2_index': explanation.roi_pairs[:, 1] if len(explanation.roi_pairs) else [],
      'importance': explanation.weights,
    })
    explanation_df.to_csv(artifact_dir / f'{method_slug}_explanation.csv', index=False)


def write_pipeline_artifacts(summary):
  if not summary.config.save_artifacts or summary.artifact_dir is None:
    return

  artifact_dir = ensure_directory(summary.artifact_dir)

  fold_metric_rows = []
  for fold_result in summary.fold_results:
    fold_metric_rows.append({
      'fold_id': fold_result.fold_id,
      'selected_feature_count': fold_result.selected_feature_count,
      'accuracy': fold_result.metrics['accuracy'],
      'sensitivity': fold_result.metrics['sensitivity'],
      'specificity': fold_result.metrics['specificity'],
      'precision': fold_result.metrics['precision'],
      'f1': fold_result.metrics['f1'],
    })

  pd.DataFrame(fold_metric_rows).to_csv(artifact_dir / 'fold_metrics.csv', index=False)

  for method_name, interpretation_summary in summary.interpretation_summary.items():
    method_slug = slugify_method_name(method_name)
    interpretation_summary['connections'].to_csv(artifact_dir / f'{method_slug}_connections.csv', index=False)
    interpretation_summary['rois'].to_csv(artifact_dir / f'{method_slug}_rois.csv', index=False)

  write_json_file(artifact_dir / 'summary.json', summary)


def train_and_eval_model(feature_vectors, labels_from_abide, pipeline='unknown', feature_indices=None, verbose=False, train_model=True, save_model=False, rfe_step=None, config=None):
  if config is None:
    config = ReanalysisConfig()

  if rfe_step is not None:
    config.rfe_step = rfe_step
  if save_model:
    config.save_model_checkpoints = True

  if config.correction_mode and not train_model:
    raise RuntimeError("Correction mode does not support loading legacy global checkpoints. Re-run the fold-specific training pipeline instead.")

  if feature_indices is None:
    raise ValueError("feature_indices must be provided for the corrected fold-specific reanalysis.")

  feature_vectors = sanitize_feature_matrix(feature_vectors)
  labels_from_abide = np.asarray(labels_from_abide, dtype=int)
  roi_lookup = prepare_feature_index_lookup(feature_indices)

  if feature_vectors.ndim != 2:
    raise ValueError("feature_vectors must be a 2D array of shape (samples, features).")

  set_random_seed(config.random_seed)
  skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)

  pipeline_artifact_dir = None
  if config.save_artifacts:
    pipeline_artifact_dir = ensure_directory(Path(config.artifact_root) / pipeline)

  fold_results = []

  for fold_id, (outer_train_indices, test_indices) in enumerate(skf.split(feature_vectors, labels_from_abide), start=1):
    if verbose:
      print(f"======================================\nFold {fold_id}\n======================================")

    outer_train_indices = np.asarray(outer_train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)

    train_indices, validation_indices = split_outer_train_validation(
      outer_train_indices,
      labels_from_abide,
      config.validation_size,
      config.random_seed + fold_id,
    )

    selected_feature_count, feature_count_tuning_records = select_feature_count_for_fold(
      feature_vectors,
      labels_from_abide,
      roi_lookup,
      train_indices,
      validation_indices,
      config,
      verbose=verbose,
    )

    selection = get_top_features_from_SVM_RFE(
      feature_vectors[outer_train_indices],
      labels_from_abide[outer_train_indices],
      roi_lookup,
      selected_feature_count,
      config.rfe_step,
      training_sample_indices=outer_train_indices,
    )

    train_features = apply_feature_selection(feature_vectors[train_indices], selection)
    validation_features = apply_feature_selection(feature_vectors[validation_indices], selection)
    test_features = apply_feature_selection(feature_vectors[test_indices], selection)
    train_features, [validation_features, test_features], scaling_summary = scale_feature_sets(
      train_features,
      config,
      validation_features,
      test_features,
    )

    train_labels = labels_from_abide[train_indices]
    validation_labels = labels_from_abide[validation_indices]
    test_labels = labels_from_abide[test_indices]

    train_dataloader = build_dataloader(train_features, train_labels, config.batch_size, shuffle=True)
    validation_dataloader = build_dataloader(validation_features, validation_labels, config.batch_size, shuffle=False)
    test_dataloader = build_dataloader(test_features, test_labels, config.batch_size, shuffle=False)

    if verbose:
      print(
        f"Fold {fold_id}: selected {selected_feature_count} features "
        f"from {feature_vectors.shape[1]} candidates"
      )

    model, training_summary = train_single_fold_model(
      train_dataloader,
      validation_dataloader,
      input_size=train_features.shape[1],
      config=config,
      verbose=verbose,
    )
    metrics, true_labels, predicted_labels = evaluate_model(model, test_dataloader)
    explanation_rankings = compute_fold_explanations(
      model,
      train_dataloader,
      test_dataloader,
      selection.selected_roi_pairs,
      config,
    )

    fold_artifact_dir = None
    model_checkpoint_path = None
    if pipeline_artifact_dir is not None:
      fold_artifact_dir = ensure_directory(pipeline_artifact_dir / f'fold_{fold_id:02d}')
      if config.save_model_checkpoints:
        model_checkpoint_path = fold_artifact_dir / 'model_state_dict.pth'
        torch.save(model.state_dict(), model_checkpoint_path)

    fold_result = FoldResult(
      fold_id=fold_id,
      outer_train_indices=outer_train_indices,
      train_indices=train_indices,
      validation_indices=validation_indices,
      test_indices=test_indices,
      selection=selection,
      selected_feature_count=selected_feature_count,
      metrics=metrics,
      true_labels=true_labels,
      predicted_labels=predicted_labels,
      train_feature_shape=train_features.shape,
      validation_feature_shape=validation_features.shape,
      test_feature_shape=test_features.shape,
      scaler_mean=np.asarray(scaling_summary['mean'], dtype=float),
      scaler_scale=np.asarray(scaling_summary['scale'], dtype=float),
      feature_count_tuning_records=feature_count_tuning_records,
      training_summary=training_summary,
      explanation_rankings=explanation_rankings,
      artifact_dir=str(fold_artifact_dir) if fold_artifact_dir is not None else None,
      model_checkpoint_path=str(model_checkpoint_path) if model_checkpoint_path is not None else None,
    )

    if fold_artifact_dir is not None:
      write_fold_artifacts(fold_result, fold_artifact_dir)

    fold_results.append(fold_result)

  metrics_summary = summarize_metrics(fold_results)
  interpretation_summary = aggregate_explanations_across_folds(fold_results, config.explanation_top_n)
  summary = PipelineRunSummary(
    pipeline=pipeline,
    config=config,
    fold_results=fold_results,
    metrics_summary=metrics_summary,
    interpretation_summary=interpretation_summary,
    artifact_dir=str(pipeline_artifact_dir) if pipeline_artifact_dir is not None else None,
  )
  write_pipeline_artifacts(summary)

  if verbose:
    print(f"Accuracy: {(summary.metrics_summary['accuracy']['mean'] * 100):.2f}%")
    print(f"Specificity: {summary.metrics_summary['specificity']['mean']:.2f}")
    print(f"Precision: {summary.metrics_summary['precision']['mean']:.2f}")
    print(f"F1_Score: {summary.metrics_summary['f1']['mean']:.2f}")

  return summary

def roar(data, labels, methods, percentiles):
  raise RuntimeError(
    "ROAR is disabled in the corrected workflow because the legacy implementation depends on a single global feature ranking. "
    "Re-run ROAR from fold-specific corrected artifacts instead."
  )


def get_accuracy_of_model_over_percentiles(percentiles, data, labels, feature_ranking, method):
  raise RuntimeError(
    "Legacy ROAR percentile retraining is disabled in correction mode. Use fold-specific corrected artifacts for any ROAR rerun."
  )


def replace_features_with_0(data, feature_indices_to_remove):
  data[:, feature_indices_to_remove] = 0

  return data


def build_hyperparameter_sweep_configs(
  base_config,
  pipeline,
  sweep_root,
  feature_counts=None,
  ae1_hidden_sizes=None,
  ae2_hidden_sizes=None,
  ae_learning_rates=None,
  classifier_learning_rates=None,
  fine_tuning_learning_rates=None,
  weight_decays=None,
):
  sweep_root = Path(sweep_root)
  feature_counts = normalize_sweep_candidates(feature_counts, base_config.num_selected_features, int)
  ae1_hidden_sizes = normalize_sweep_candidates(ae1_hidden_sizes, base_config.ae1_hidden_size or 1, int)
  ae2_hidden_sizes = normalize_sweep_candidates(ae2_hidden_sizes, base_config.ae2_hidden_size or 1, int)
  ae_learning_rates = normalize_sweep_candidates(ae_learning_rates, base_config.ae_learning_rate, float)
  classifier_learning_rates = normalize_sweep_candidates(classifier_learning_rates, base_config.classifier_learning_rate, float)
  fine_tuning_learning_rates = normalize_sweep_candidates(fine_tuning_learning_rates, base_config.fine_tuning_learning_rate, float)
  weight_decays = normalize_sweep_candidates(weight_decays, base_config.weight_decay, float)

  sweep_configs = []

  for combination_index, combination in enumerate(product(
    feature_counts,
    ae1_hidden_sizes,
    ae2_hidden_sizes,
    ae_learning_rates,
    classifier_learning_rates,
    fine_tuning_learning_rates,
    weight_decays,
  ), start=1):
    (
      feature_count,
      ae1_hidden_size,
      ae2_hidden_size,
      ae_learning_rate,
      classifier_learning_rate,
      fine_tuning_learning_rate,
      weight_decay,
    ) = combination

    config_name = (
      f"fs{feature_count}"
      f"_ae1{slugify_sweep_value(ae1_hidden_size)}"
      f"_ae2{slugify_sweep_value(ae2_hidden_size)}"
      f"_aelr{slugify_sweep_value(ae_learning_rate)}"
      f"_clrlr{slugify_sweep_value(classifier_learning_rate)}"
      f"_ftlr{slugify_sweep_value(fine_tuning_learning_rate)}"
      f"_wd{slugify_sweep_value(weight_decay)}"
    )

    config = ReanalysisConfig(**asdict(base_config))
    config.num_selected_features = int(feature_count)
    config.feature_count_candidates = ()
    config.ae1_hidden_size = int(ae1_hidden_size)
    config.ae2_hidden_size = int(ae2_hidden_size)
    config.ae_learning_rate = float(ae_learning_rate)
    config.classifier_learning_rate = float(classifier_learning_rate)
    config.fine_tuning_learning_rate = float(fine_tuning_learning_rate)
    config.weight_decay = float(weight_decay)
    config.explanation_methods = ()
    config.save_model_checkpoints = False
    config.artifact_root = str(sweep_root / config_name)

    sweep_configs.append({
      'index': combination_index,
      'name': config_name,
      'parameters': {
        'num_selected_features': int(feature_count),
        'ae1_hidden_size': int(ae1_hidden_size),
        'ae2_hidden_size': int(ae2_hidden_size),
        'ae_learning_rate': float(ae_learning_rate),
        'classifier_learning_rate': float(classifier_learning_rate),
        'fine_tuning_learning_rate': float(fine_tuning_learning_rate),
        'weight_decay': float(weight_decay),
      },
      'config': config,
      'pipeline': pipeline,
    })

  return sweep_configs


def run_hyperparameter_sweep(pipeline, base_config, sweep_configs, sweep_root, verbose=False):
  sweep_root = ensure_directory(sweep_root)
  sweep_rows = []
  sweep_records = []

  for sweep_entry in sweep_configs:
    config_name = sweep_entry['name']
    config = sweep_entry['config']

    if verbose:
      print("\n" + "-" * 100)
      print(f"Sweep configuration {sweep_entry['index']}/{len(sweep_configs)}: {config_name}")
      print(json.dumps(sweep_entry['parameters'], indent=2))
      print("-" * 100)

    summary = run_pipeline_reanalysis(pipeline, verbose=verbose, config=config)

    row = {
      'config_name': config_name,
      **sweep_entry['parameters'],
      'accuracy_mean': float(summary.metrics_summary['accuracy']['mean']),
      'accuracy_std': float(summary.metrics_summary['accuracy']['std']),
      'f1_mean': float(summary.metrics_summary['f1']['mean']),
      'f1_std': float(summary.metrics_summary['f1']['std']),
      'specificity_mean': float(summary.metrics_summary['specificity']['mean']),
      'precision_mean': float(summary.metrics_summary['precision']['mean']),
      'artifact_dir': summary.artifact_dir,
      'fold_selected_feature_counts': [int(fold_result.selected_feature_count) for fold_result in summary.fold_results],
    }
    sweep_rows.append(row)
    sweep_records.append({
      'config_name': config_name,
      'parameters': sweep_entry['parameters'],
      'summary': summary,
    })

  sweep_df = pd.DataFrame(sweep_rows)
  if not sweep_df.empty:
    sweep_df = sweep_df.sort_values(
      ['accuracy_mean', 'f1_mean', 'accuracy_std'],
      ascending=[False, False, True],
    ).reset_index(drop=True)

  sweep_df.to_csv(sweep_root / 'results.csv', index=False)
  write_json_file(sweep_root / 'results.json', sweep_rows)

  return sweep_df, sweep_records


def run_pipeline_reanalysis(pipeline, verbose=False, config=None):
  data, labels = get_data_from_abide(pipeline)
  feature_vectors, feature_indices = get_feature_vecs(data)
  return train_and_eval_model(
    feature_vectors,
    labels,
    pipeline=pipeline,
    feature_indices=feature_indices,
    verbose=verbose,
    train_model=True,
    save_model=False,
    rfe_step=config.rfe_step if config is not None else None,
    config=config,
  )


def run_permutation_test(feature_vectors, labels, feature_indices, pipeline, config, n_permutations=100, verbose=False):
  rng = np.random.default_rng(config.random_seed)
  observed_summary = train_and_eval_model(
    feature_vectors,
    labels,
    pipeline=pipeline,
    feature_indices=feature_indices,
    verbose=verbose,
    train_model=True,
    save_model=False,
    rfe_step=config.rfe_step,
    config=config,
  )

  permutation_accuracies = []
  permutation_f1_scores = []

  permutation_config = ReanalysisConfig(**asdict(config))
  permutation_config.save_artifacts = False
  permutation_config.save_model_checkpoints = False

  for permutation_index in range(n_permutations):
    permuted_labels = rng.permutation(labels)
    permutation_summary = train_and_eval_model(
      feature_vectors,
      permuted_labels,
      pipeline=f"{pipeline}_permutation_{permutation_index + 1:03d}",
      feature_indices=feature_indices,
      verbose=False,
      train_model=True,
      save_model=False,
      rfe_step=permutation_config.rfe_step,
      config=permutation_config,
    )
    permutation_accuracies.append(permutation_summary.metrics_summary['accuracy']['mean'])
    permutation_f1_scores.append(permutation_summary.metrics_summary['f1']['mean'])

  observed_accuracy = observed_summary.metrics_summary['accuracy']['mean']
  observed_f1 = observed_summary.metrics_summary['f1']['mean']

  permutation_results = {
    'observed_accuracy': float(observed_accuracy),
    'observed_f1': float(observed_f1),
    'null_accuracies': [float(value) for value in permutation_accuracies],
    'null_f1_scores': [float(value) for value in permutation_f1_scores],
    'accuracy_p_value': float((1 + np.sum(np.asarray(permutation_accuracies) >= observed_accuracy)) / (n_permutations + 1)),
    'f1_p_value': float((1 + np.sum(np.asarray(permutation_f1_scores) >= observed_f1)) / (n_permutations + 1)),
  }

  if config.save_artifacts:
    artifact_dir = ensure_directory(Path(config.artifact_root) / pipeline)
    write_json_file(artifact_dir / 'permutation_test.json', permutation_results)

  return observed_summary, permutation_results

def compare_models_in_different_pipelines():
  raise RuntimeError(
    "Use the corrected per-pipeline summaries produced by train_and_eval_model or run_pipeline_reanalysis. "
    "The legacy cross-pipeline comparison relied on globally selected features."
  )

def overlap_coefficient(set_a, set_b):
    # Calculate the intersection and the sizes of the sets
    # intersection = len(get_relaxed_overlap(set_a, set_b))
    set_a = set(set_a)
    set_b = set(set_b)

    intersection = len(set_a.intersection(set_b))
    min_size = min(len(set_a), len(set_b))
    
    # Calculate the Overlap Coefficient
    overlap_coef = intersection / min_size
    return overlap_coef


def calculate_similarity_metrics(connections_a, connections_b):
  overlap_coef = overlap_coefficient(connections_a, connections_b)

  # Convert connection lists to binary vectors
  connections_a = set(connections_a)
  connections_b = set(connections_b)
  all_connections = list(set(connections_a) | set(connections_b))
  vector_a = np.array([1 if conn in connections_a else 0 for conn in all_connections])
  vector_b = np.array([1 if conn in connections_b else 0 for conn in all_connections])

  # Jaccard Index
  jaccard = jaccard_score(vector_a, vector_b)
  
  # Cosine Similarity
  cosine_sim = cosine_similarity([vector_a], [vector_b])[0][0]
  
  return jaccard, cosine_sim, overlap_coef

def calculate_spatial_overlap(connections1, connections2):
  roi_set1 = set(connections1)
  roi_set2 = set(connections2)
  overlap = roi_set1.intersection(roi_set2)
  overlap_fraction = len(overlap) / len(roi_set1.union(roi_set2))

  return overlap_fraction

def get_base_name(roi):
  return roi.split('_')[0]

def get_relaxed_overlap(rois_1, rois_2, centroid_distance_threshold=0.5):
  # base_rois_1 = {get_base_name(roi) for roi in rois_1}
  # base_rois_2 = {get_base_name(roi) for roi in rois_2}

  # relaxed_overlap = base_rois_1.intersection(base_rois_2)

  # # Find the original ROIs corresponding to the overlapping base names
  # overlap = {roi for roi in rois_1 if get_base_name(roi) in relaxed_overlap} | {roi for roi in rois_2 if get_base_name(roi) in relaxed_overlap}

  atlas = datasets.fetch_atlas_aal(version='SPM12')
  labels = atlas.labels  # List of AAL region labels
  label_indices = {label: index for index, label in enumerate(labels)}

  # Get the indices of the selected ROIs
  selected_indices = [label_indices[roi] for roi in rois_1 if roi in label_indices]
  selected_labels = [labels[index] for index in selected_indices]

  # Get the coordinates of the ROIs
  coordinates = expand_relative_coords(plotting.find_parcellation_cut_coords(atlas.maps), 1.08) 
  filtered_coordinates = [coordinates[index] for index in selected_indices]

  overlap = []

  return overlap

def compare_pipelines(pipeline1, pipeline2, strict=True, method='Integrated Gradients', artifact_root='artifacts/reanalysis'):
  rois_1 = get_top_rois(pipeline1, method=method, artifact_root=artifact_root)['roi_index']
  rois_2 = get_top_rois(pipeline2, method=method, artifact_root=artifact_root)['roi_index']

  print(f"Top ROI indices for {pipeline1}: {set(rois_1)}")
  print(f"Top ROI indices for {pipeline2}: {set(rois_2)}")

  spatial_overlap = calculate_spatial_overlap(rois_1, rois_2)

  if strict:
    overlap = set(rois_1).intersection(set(rois_2))
  else:
    overlap = set(rois_1).intersection(set(rois_2))

  print(f"Relaxed Overlap between {pipeline1} and {pipeline2}: {overlap}")

  jaccard, cosine_sim, overlap_coef = calculate_similarity_metrics(rois_1, rois_2)

  print(f"Similarity between {pipeline1} and {pipeline2}:")
  print(f"Jaccard Index: {jaccard}")
  print(f"Cosine Similarity: {cosine_sim}")
  print(f"Overlap Coefficient: {overlap_coef}")
  print(f"Spatial Overlap: {spatial_overlap}")

  return overlap


def get_top_rois(pipeline_or_summary, labels_from_abide=None, RFE_step=20, N_rois=1000, method='Integrated Gradients', artifact_root='artifacts/reanalysis'):
  if isinstance(pipeline_or_summary, PipelineRunSummary):
    interpretation_summary = pipeline_or_summary.interpretation_summary.get(method)
    if interpretation_summary is None:
      raise KeyError(f"No interpretation summary stored for method '{method}'.")
    return interpretation_summary['rois'].head(N_rois).copy()

  pipeline = str(pipeline_or_summary)
  method_slug = slugify_method_name(method)
  artifact_path = Path(artifact_root) / pipeline / f'{method_slug}_rois.csv'
  if not artifact_path.exists():
    raise FileNotFoundError(
      f"Could not find corrected ROI summary at '{artifact_path}'. "
      "Run the corrected reanalysis first; legacy global ROI tables are disabled."
    )
  return pd.read_csv(artifact_path).head(N_rois)


def print_rois(rois):
  # Fetch the AAL atlas
  atlas = datasets.fetch_atlas_aal(version='SPM12')
  labels = atlas.labels  # List of AAL region labels
  label_indices = {label: index for index, label in enumerate(labels)}

  # Get the indices of the selected ROIs
  selected_indices = [label_indices[roi] for roi in rois if roi in label_indices]
  selected_labels = [labels[index] for index in selected_indices]

  # Get the coordinates of the ROIs
  coordinates = expand_relative_coords(plotting.find_parcellation_cut_coords(atlas.maps), 1.08) 
  filtered_coordinates = [coordinates[index] for index in selected_indices]

  # Create an empty adjacency matrix for all ROIs
  full_adjacency_matrix = np.zeros((len(labels), len(labels)))

  # Plot all nodes with a default color and size
  fig, ax = plt.subplots(figsize=(10, 8))
  plotting.plot_connectome(full_adjacency_matrix, coordinates,
                            node_color='darkblue',
                            node_size=25,
                            display_mode='ortho',
                            colorbar=False,
                            title="Selected ROIs",
                            axes=ax)

  # Highlight selected nodes with a different color and size
  selected_coordinates = [coordinates[index] for index in selected_indices]
  selected_adjacency_matrix = np.zeros((len(selected_indices), len(selected_indices)))
  plotting.plot_connectome(selected_adjacency_matrix, selected_coordinates,
                            node_color='yellow',
                            node_size=100,
                            display_mode='ortho',
                            colorbar=False,
                            title="Selected ROIs",
                            axes=ax)

  # Overlay labels for the highlighted nodes
  for coord, label in zip(selected_coordinates, selected_labels):
      x, y, z = coord
      ax.text(x, y, label, fontsize=9, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

  plt.show()

def view_rois(rois):
  # Fetch the AAL atlas
  atlas = datasets.fetch_atlas_aal(version='SPM12')
  labels = atlas.labels  # List of AAL region labels
  label_indices = {label: index for index, label in enumerate(labels)}

  # Filter to include only the requested ROIs
  filtered_indices = [label_indices[roi] for roi in rois if roi in label_indices]
  filtered_labels = [labels[index] for index in filtered_indices]

  # Get the coordinates of the ROIs
  coordinates = plotting.find_parcellation_cut_coords(atlas.maps)
  filtered_coordinates = [coordinates[index] for index in filtered_indices]
  
  plotting_params = {
    'marker_color': [],
    'marker_size': [],
    'marker_labels': [],
    'coordinates': coordinates
  }

  for idx, coord in enumerate(coordinates):
    if np.any(filtered_coordinates == coord):
      plotting_params['marker_color'].append('yellow')
      plotting_params['marker_size'].append(25)
      plotting_params['marker_labels'].append(labels[idx])
    else:
      plotting_params['marker_color'].append('darkblue')
      plotting_params['marker_size'].append(10)
      plotting_params['marker_labels'].append('')

  # Plot all coordinates with view_markers
  all_markers = plotting.view_markers(plotting_params['coordinates'], marker_labels=plotting_params['marker_labels'], marker_color=plotting_params['marker_color'], marker_size=plotting_params['marker_size'])

  # Show both plots in the browser
  all_markers.open_in_browser()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process control flags.')
  parser.add_argument('--verbose', type=lambda x: (str(x).lower() == 'true'), default=False, help='Verbose output')
  parser.add_argument('--train_model', type=lambda x: (str(x).lower() == 'true'), default=True, help='Train model')
  parser.add_argument('--save_model', type=lambda x: (str(x).lower() == 'true'), default=False, help='Save the trained model')
  parser.add_argument('--interpretation_methods', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use interpretation methods')
  parser.add_argument('--analyze_methods', type=lambda x: (str(x).lower() == 'true'), default=False, help='Legacy flag retained for CLI compatibility. ROAR is disabled in correction mode.')
  parser.add_argument('--pipelines', nargs='*', default=['ccs', 'cpac', 'dparsf', 'niak'], help='Pipelines to reanalyse using fold-specific feature selection.')
  parser.add_argument('--artifact_root', default='artifacts/reanalysis', help='Directory for corrected reanalysis artifacts.')
  parser.add_argument('--num_selected_features', type=int, default=1000, help='Number of fold-local SVM-RFE features to keep.')
  parser.add_argument('--ae1_hidden_size', type=int, default=500, help='Hidden size for the first autoencoder layer.')
  parser.add_argument('--ae2_hidden_size', type=int, default=100, help='Hidden size for the second autoencoder layer.')
  parser.add_argument('--feature_count_candidates', nargs='*', type=int, default=None, help='Optional feature-count candidates to tune inside each training fold, e.g. --feature_count_candidates 250 500 750 1000')
  parser.add_argument('--feature_count_selection_metric', default='f1', help='Validation metric used to choose feature count. Options: accuracy, sensitivity, specificity, precision, f1')
  parser.add_argument('--rfe_step', type=int, default=20, help='SVM-RFE elimination step size.')
  parser.add_argument('--early_stopping_patience', type=int, default=15, help='Number of validation epochs without improvement before stopping classifier/fine-tuning.')
  parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4, help='Minimum validation-loss improvement required to reset early stopping.')
  parser.add_argument('--ae_learning_rate', type=float, default=0.001, help='Learning rate for autoencoder pretraining.')
  parser.add_argument('--classifier_learning_rate', type=float, default=0.001, help='Learning rate for the classifier stage.')
  parser.add_argument('--fine_tuning_learning_rate', type=float, default=0.0001, help='Learning rate for the end-to-end fine-tuning stage.')
  parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay used across the training stages.')
  parser.add_argument('--use_feature_scaling', type=lambda x: (str(x).lower() == 'true'), default=True, help='Scale selected features inside each fold using training data only.')
  parser.add_argument('--run_hyperparameter_sweep', type=lambda x: (str(x).lower() == 'true'), default=False, help='Run a series of fixed, leak-free configurations and save a comparison table.')
  parser.add_argument('--sweep_name', default='hyperparameter_sweep', help='Artifact subdirectory name for the fixed-configuration sweep summary.')
  parser.add_argument('--sweep_feature_counts', nargs='*', type=int, default=None, help='Feature-count values to compare across full leak-free runs.')
  parser.add_argument('--sweep_ae1_hidden_sizes', nargs='*', type=int, default=None, help='AE1 hidden sizes to compare across full leak-free runs.')
  parser.add_argument('--sweep_ae2_hidden_sizes', nargs='*', type=int, default=None, help='AE2 hidden sizes to compare across full leak-free runs.')
  parser.add_argument('--sweep_ae_learning_rates', nargs='*', type=float, default=None, help='Autoencoder learning rates to compare across full leak-free runs.')
  parser.add_argument('--sweep_classifier_learning_rates', nargs='*', type=float, default=None, help='Classifier learning rates to compare across full leak-free runs.')
  parser.add_argument('--sweep_fine_tuning_learning_rates', nargs='*', type=float, default=None, help='Fine-tuning learning rates to compare across full leak-free runs.')
  parser.add_argument('--sweep_weight_decays', nargs='*', type=float, default=None, help='Weight-decay values to compare across full leak-free runs.')
  parser.add_argument('--max_sweep_configs', type=int, default=24, help='Safety cap on the number of fixed-configuration sweep combinations.')
  parser.add_argument('--run_permutation_test', type=lambda x: (str(x).lower() == 'true'), default=False, help='Run the corrected CCS permutation test.')
  parser.add_argument('--num_permutations', type=int, default=100, help='Number of label permutations for the corrected CCS permutation test.')

  args = parser.parse_args()

  verbose = args.verbose
  train_model = args.train_model
  save_model = args.save_model
  interpretation_methods = args.interpretation_methods
  analyze_methods = args.analyze_methods
  run_hyperparameter_sweep_flag = args.run_hyperparameter_sweep

  print("verbose: ", verbose)
  print("train_model: ", train_model)
  print("save_model: ", save_model)
  print("interpretation_methods: ", interpretation_methods)
  print("analyze_methods: ", analyze_methods)
  print("run_hyperparameter_sweep: ", run_hyperparameter_sweep_flag)
  print("Torch Cuda is Available =", use_cuda)

  if not train_model:
    raise RuntimeError("Correction mode requires fold-specific retraining. Legacy checkpoint loading is disabled.")
  if analyze_methods:
    raise RuntimeError("ROAR is disabled in the corrected workflow. Re-run ROAR from fold-specific corrected artifacts instead.")
  if run_hyperparameter_sweep_flag and args.run_permutation_test:
    raise RuntimeError("Run the hyperparameter sweep and the permutation test separately to keep artifacts and runtime manageable.")

  selected_methods = DEFAULT_INTERPRETATION_METHODS if interpretation_methods else ()
  base_config = ReanalysisConfig(
    random_seed=DEFAULT_SEED,
    artifact_root=args.artifact_root,
    num_selected_features=args.num_selected_features,
    ae1_hidden_size=args.ae1_hidden_size,
    ae2_hidden_size=args.ae2_hidden_size,
    feature_count_candidates=tuple(args.feature_count_candidates or ()),
    feature_count_selection_metric=args.feature_count_selection_metric,
    rfe_step=args.rfe_step,
    early_stopping_patience=args.early_stopping_patience,
    early_stopping_min_delta=args.early_stopping_min_delta,
    ae_learning_rate=args.ae_learning_rate,
    classifier_learning_rate=args.classifier_learning_rate,
    fine_tuning_learning_rate=args.fine_tuning_learning_rate,
    weight_decay=args.weight_decay,
    use_feature_scaling=args.use_feature_scaling,
    explanation_methods=tuple(selected_methods),
    save_artifacts=True,
    save_model_checkpoints=save_model,
  )

  pipeline_summaries = {}

  for pipeline in args.pipelines:
    if run_hyperparameter_sweep_flag:
      sweep_root = ensure_directory(Path(args.artifact_root) / 'hyperparameter_sweeps' / pipeline / args.sweep_name)
      sweep_base_config = ReanalysisConfig(**asdict(base_config))
      sweep_configs = build_hyperparameter_sweep_configs(
        sweep_base_config,
        pipeline=pipeline,
        sweep_root=sweep_root,
        feature_counts=args.sweep_feature_counts,
        ae1_hidden_sizes=args.sweep_ae1_hidden_sizes,
        ae2_hidden_sizes=args.sweep_ae2_hidden_sizes,
        ae_learning_rates=args.sweep_ae_learning_rates,
        classifier_learning_rates=args.sweep_classifier_learning_rates,
        fine_tuning_learning_rates=args.sweep_fine_tuning_learning_rates,
        weight_decays=args.sweep_weight_decays,
      )

      if len(sweep_configs) > args.max_sweep_configs:
        raise RuntimeError(
          f"Requested {len(sweep_configs)} sweep configurations, which exceeds the safety cap of {args.max_sweep_configs}. "
          "Reduce the candidate lists or raise --max_sweep_configs deliberately."
        )

      print(f"\nRunning leak-free hyperparameter sweep for pipeline '{pipeline}' with {len(sweep_configs)} configurations")
      sweep_df, _ = run_hyperparameter_sweep(
        pipeline,
        sweep_base_config,
        sweep_configs,
        sweep_root,
        verbose=verbose,
      )

      if sweep_df.empty:
        print(f"No sweep results were produced for pipeline '{pipeline}'.")
      else:
        print("\nBest sweep results\n")
        print(sweep_df.head(10).to_string(index=False))
      continue

    print(f"\nRunning corrected reanalysis for pipeline '{pipeline}'")
    pipeline_config = ReanalysisConfig(**asdict(base_config))
    summary = run_pipeline_reanalysis(pipeline, verbose=verbose, config=pipeline_config)
    pipeline_summaries[pipeline] = summary

    print(
      f"{pipeline}: accuracy={summary.metrics_summary['accuracy']['mean']:.4f} ± {summary.metrics_summary['accuracy']['std']:.4f}, "
      f"f1={summary.metrics_summary['f1']['mean']:.4f} ± {summary.metrics_summary['f1']['std']:.4f}"
    )

    if interpretation_methods:
      for method_name, interpretation_summary in summary.interpretation_summary.items():
        print("\n" + "=" * 100)
        print(method_name)
        print("=" * 100)
        if interpretation_summary['skipped_folds']:
          print(f"Skipped folds: {interpretation_summary['skipped_folds']}")
        print("\nTop Connections\n")
        print(interpretation_summary['connections'].head(10).to_string(index=False))
        print("\nTop ROIs\n")
        print(interpretation_summary['rois'].head(10).to_string(index=False))

  if args.run_permutation_test:
    print("\nRunning corrected CCS permutation test")
    ccs_data, ccs_labels = get_data_from_abide('ccs')
    ccs_feature_vectors, ccs_feature_indices = get_feature_vecs(ccs_data)
    permutation_config = ReanalysisConfig(**asdict(base_config))
    permutation_config.explanation_methods = ()
    observed_summary, permutation_results = run_permutation_test(
      ccs_feature_vectors,
      ccs_labels,
      ccs_feature_indices,
      pipeline='ccs',
      config=permutation_config,
      n_permutations=args.num_permutations,
      verbose=verbose,
    )
    pipeline_summaries['ccs'] = observed_summary
    print(json.dumps(permutation_results, indent=2))

  print("Seed is", DEFAULT_SEED)
