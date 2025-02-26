""" 
Linear probe evaluation script that trains a linear classifier on frozen embeddings.

Config file should be a YAML with the following structure:

Example config:
```yaml
# Wandb logging settings
wandb_project: "clip-mimic-linear-probe"
run_name: "clip-mimic-wbce"

# Basic settings
dataset: "mimic"              # Dataset name (mimic, nih, chexpert, vindr-cxr, mimic_lt, nih_lt)
model: "clip"                 # Model name used in embedding filenames
multilabel: true             # Whether dataset is multilabel or single-label
num_classes: 14              # Number of classes

# Paths
base_path: "data/"           # Base path containing embeddings and labels
label_file: "mimic-cxr-2.0.0-merged.csv"

# Training parameters
batch_size: 1350
learning_rate: 3e-5
num_epochs: 15
warmup_epochs: 2

# Loss function - options:
# Multilabel: ["bce", "wbce", "asl", "w-asl", "DBLoss", "ral"]
# Single-label: ["ce", "wce"]
loss: "wbce"

# Optional parameters
seed: 1234
num_workers: 12

```

"""

import torch
import torch.nn as nn
import torch.optim as optim
from optimi import AdamW
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
import wandb
import yaml
import argparse
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score

from metrics import validate_multilabel, validate_singlelabel, init_wandb, log_training_step, log_epoch_metrics, log_test_results
from losses import get_multilabel_loss, get_singlelabel_loss 
from ml_decoder import MLDecoder

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes, head_type="linear"):
        super().__init__()
        if head_type == "linear":
            self.classifier = nn.Linear(input_dim, num_classes)
        elif head_type == "MLDecoder":
            self.classifier = MLDecoder(
                num_classes=num_classes,
                initial_num_features=input_dim,
                num_of_groups=-1,
                decoder_embedding=768,  # Fixed embedding dimension
                zsl=0
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")
        self.head_type = head_type

    def forward(self, x):
        if self.head_type == "MLDecoder":
            # MLDecoder expects input shape (batch_size, num_tokens, embedding_dim)
            # So we need to add a dimension
            x = x.unsqueeze(1)
        return self.classifier(x)

def setup_logging(args, rank=0):
    if rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Create runs directory if it doesn't exist
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        
        # Create run-specific directory
        log_dir = runs_dir / args['run_name']
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"{timestamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        return logging.getLogger(__name__)
    return None

def get_criterion(args, train_labels):
    """Setup loss function based on config"""
    total_instance_num = len(train_labels)
    if args['multilabel']:
        class_instance_nums = train_labels.sum(dim=0).tolist()
        return get_multilabel_loss(args['loss'], class_instance_nums=class_instance_nums, total_instance_num=total_instance_num)
    else:
        class_instance_nums = torch.bincount(train_labels)
        return get_singlelabel_loss(args['loss'], class_instance_nums=class_instance_nums, total_instance_num=total_instance_num)

def load_embeddings(base_path, dataset_name, model_name, split, representation="both"):
    """Load embeddings for a specific dataset, model and split"""
    # Handle special case for vindr dataset names in file paths
    file_dataset_name = dataset_name.replace("-", "-")  # Keep as is for normal cases
    # if dataset_name in ['vindr-cxr']:
        # file_dataset_name = dataset_name.replace("-", "-")
    if dataset_name in ["mimic_lt", "nih_lt"]:
        file_dataset_name = dataset_name.replace("_", "-")

    if dataset_name in ['rsna-pneumonia', 'siim-pneumothorax']:
        file_dataset_name = dataset_name.replace("-", "_")
    
    # For CheXpert, use validation split instead of test
    if dataset_name == "chexpert" and split == "test":
        split = "val"
    
    # Map model names to their corresponding folders and embedding types
    model_folder_map = {
        "bert": "bert-single",
        "cxr_bert": "cxr-bert-dual",
        "scratch": "scratch-transformer-single",
        "dino": "dino"
    }
    
    model_folder = model_folder_map.get(model_name)
    if not model_folder:
        raise ValueError(f"Unknown model type: {model_name}. Must be one of: {list(model_folder_map.keys())}")
    
    # Construct the embedding folder name based on model type
    if model_folder == "dino":
        embedding_folder = f"{file_dataset_name}-cxrbert-dual-embeddings"
        embedding_prefix = "dino"
    elif model_folder == "bert-single":
        embedding_folder = f"{file_dataset_name}-bert-single-embeddings"
        embedding_prefix = "clip"
    elif model_folder == "cxr-bert-dual":
        embedding_folder = f"{file_dataset_name}-cxrbert-dual-embeddings"
        embedding_prefix = "clip"
    elif model_folder == "scratch-transformer-single":
        embedding_folder = f"{file_dataset_name}-scratch-transformer-single-embeddings"
        embedding_prefix = "clip"
    else:
        embedding_folder = f"{file_dataset_name}-embeddings"
        embedding_prefix = "clip"
    
    file_dataset_name = file_dataset_name.replace("-", "_")
    if "vindr" in file_dataset_name:
        file_dataset_name = file_dataset_name.replace("_", "-")

    # Construct the embedding path
    embedding_path = os.path.join(
        base_path,
        "embeddings",
        model_folder,
        embedding_folder,
        f"{file_dataset_name}_{embedding_prefix}_{split}_embeddings.npy"
    )
    
    embeddings = np.load(embedding_path)
    
    # Handle different representation types
    if representation == "cls":
        embeddings = embeddings[:, :768]  # Take first 768 dimensions
    elif representation == "patch_mean":
        embeddings = embeddings[:, 768:]  # Take last 768 dimensions
    # For "both", use full embeddings
    
    return embeddings

def get_dataset_config():
    """Get configuration for datasets"""
    return {
        "mimic": {
            "label_file": "mimic-cxr-2.0.0-merged-with-paths.csv",
            "columns": ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                       'Lung Opacity', 'No Finding', 'Pleural Effusion',
                       'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'],
            "multilabel": True
        },
        "mimic_lt": {
            "label_file": "mimic-cxr-lt-merged.csv",
            "columns": ['Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly', 
                       'Consolidation', 'Edema', 'Emphysema', 'Enlarged Cardiomediastinum', 
                       'Fibrosis', 'Fracture', 'Hernia', 'Infiltration', 'Lung Lesion', 
                       'Lung Opacity', 'Mass', 'No Finding', 'Nodule', 'Pleural Effusion', 
                       'Pleural Other', 'Pleural Thickening', 'Pneumomediastinum', 
                       'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 
                       'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta']
        },
        "chexpert": {
            "label_file": "chexpert_plus_labels_with_5x200.csv",
            "columns": ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                       'Support Devices', 'No Finding']
        },
        "nih": {
            "label_file": "nih-merged.csv",
            "columns": ['Effusion', 'Cardiomegaly', 'Hernia', 'Infiltration', 'Mass',
                       'Pneumonia', 'Pleural_Thickening', 'No Finding', 'Edema', 'Nodule',
                       'Pneumothorax', 'Consolidation', 'Fibrosis', 'Atelectasis', 'Emphysema']
        },
        "nih_lt": {
            "label_file": "nih-lt-merged.csv",
            "columns": ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
                       'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
                       'Pleural Thickening', 'Pneumonia', 'Pneumothorax', 'Pneumoperitoneum',
                       'Pneumomediastinum', 'Subcutaneous Emphysema', 'Tortuous Aorta',
                       'Calcification of the Aorta', 'No Finding']
        },
        "vindr-cxr": {
            "label_file": "vin-dr-cxr-merged.csv",
            "columns": ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
                       'Clavicle fracture', 'Consolidation', 'Emphysema', 'Enlarged PA',
                       'ILD', 'Infiltration', 'Lung Opacity', 'Lung cavity', 'Lung cyst',
                       'Mediastinal shift', 'Nodule/Mass', 'Pleural effusion', 'Pleural thickening',
                       'Pneumothorax', 'Pulmonary fibrosis', 'Rib fracture', 'COPD', 'Lung tumor',
                       'Pneumonia', 'Tuberculosis', 'No finding']
        },
        "covid": {
            "label_file": "covid-merged.csv",
            "num_classes": 2,  # Specify number of classes for single-label
            "multilabel": False
        },
        "rsna-pneumonia": {
            "label_file": "rsna-pneumonia-merged.csv",
            "num_classes": 2,
            "multilabel": False
        },
        "siim-pneumothorax": {
            "label_file": "siim-pneumothorax-merged.csv",
            "num_classes": 2,
            "multilabel": False
        },
        "tbx11k": {
            "label_file": "tbx11k-merged.csv",
            "num_classes": 3,
            "multilabel": False
        },
        "chexchonet": {
            "label_file": "chexchonet-merged.csv",
            "num_classes": 2,
            "multilabel": False
        }
    }

def categorize_classes(train_labels):
    """
    Categorize classes into head, medium and tail based on their frequency.
    
    Args:
        train_labels: numpy array of training labels
        
    Returns:
        dict: Maps category (head/medium/tail) to list of class indices
    """
    if len(train_labels.shape) == 1:  # Single label case
        class_frequencies = np.bincount(train_labels) / len(train_labels)
    else:  # Multi label case
        class_frequencies = train_labels.mean(axis=0)
    
    categories = {
        'head': [],
        'medium': [],
        'tail': []
    }
    
    for i, freq in enumerate(class_frequencies):
        if freq > 0.15:  # >15%
            categories['head'].append(i)
        elif freq > 0.03:  # 3-15%
            categories['medium'].append(i)
        else:  # <3%
            categories['tail'].append(i)
            
    return categories

def load_and_process_labels(base_path, dataset_name, multilabel=True):
    """Load and process labels dataframe"""
    dataset_configs = get_dataset_config()
    config = dataset_configs[dataset_name]
    
    # Override multilabel parameter with dataset config
    multilabel = config.get('multilabel', multilabel)
    
    labels_path = os.path.join(base_path, "labels", config['label_file'])
    labels = pd.read_csv(labels_path, sep='\t')
    
    # For CheXpert, use validation split instead of test
    if dataset_name == "chexpert":
        train_labels = labels[labels['split'] == 'train']
        test_labels = labels[labels['split'] == 'valid']
    else:
        train_labels = labels[labels['split'] == 'train']
        test_labels = labels[labels['split'] == 'test']
    
    if multilabel:
        # Use predefined columns from config
        label_columns = config['columns']
        
        # Select only the disease columns
        train_labels = train_labels[label_columns]
        test_labels = test_labels[label_columns]
        
        # Replace -1 with 1 and NaN with 0 for multilabel
        train_labels = train_labels.replace(-1, 1).fillna(0)
        test_labels = test_labels.replace(-1, 1).fillna(0)
    else:
        # For single label, just get the 'label' column
        train_labels = train_labels['label']
        test_labels = test_labels['label']
        label_columns = None  # No label columns for single-label
    
    return (train_labels.to_numpy(), test_labels.to_numpy(), label_columns)

def save_results_to_csv(metrics, config, output_dir=None):
    """Save final results to CSV files - one for overall metrics and one for per-class metrics"""
    # Use runs/run-name as the output directory if not specified
    if output_dir is None:
        output_dir = os.path.join("runs", config['run_name'])

    multilabel = config['multilabel']
    dataset = config['dataset']
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine filenames based on multilabel flag and dataset
    if multilabel:
        overall_file = f"{output_dir}/overall_metrics_multilabel.csv"
        per_class_file = f"{output_dir}/per_class_metrics_multilabel.csv"
    else:
        # Map dataset names to their specific filenames
        dataset_file_map = {
            'covid': ('overall_metrics_covid.csv', 'per_class_metrics_covid.csv'),
            'rsna-pneumonia': ('overall_metrics_rsna-pneumonia.csv', 'per_class_metrics_rsna-pneumonia.csv'),
            'chexchonet': ('overall_metrics_chexchonet.csv', 'per_class_metrics_chexchonet.csv'),
            'siim-pneumothorax': ('overall_metrics_siim-pneumothorax.csv', 'per_class_metrics_siim-pneumothorax.csv'),
            'tbx11k': ('overall_metrics_tbx11k.csv', 'per_class_metrics_tbx11k.csv')
        }
        overall_file = f"{output_dir}/{dataset_file_map.get(dataset, ('overall_metrics.csv', 'per_class_metrics.csv'))[0]}"
        per_class_file = f"{output_dir}/{dataset_file_map.get(dataset, ('overall_metrics.csv', 'per_class_metrics.csv'))[1]}"
    
    # Prepare data for overall metrics CSV - only final values
    overall_row = {
        'dataset': config['dataset'],
        'model': config['model'],
        'roc_auc': metrics['roc_auc'],
        'average_precision': metrics['average_precision']
    }
    
    # Add precision, recall, f1 for single-label classification
    if not multilabel:
        overall_row.update({
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
    
    # Add category metrics to overall_row
    # Add all ROC AUC metrics first
    for category in ['head', 'medium', 'tail']:
        roc_key = f'{category}_roc_auc'
        if roc_key in metrics:
            overall_row[roc_key] = metrics[roc_key]
    
    # Then add all average precision metrics
    for category in ['head', 'medium', 'tail']:
        ap_key = f'{category}_avg_precision'
        if ap_key in metrics:
            overall_row[ap_key] = metrics[ap_key]
    
    # Add per-class metrics to overall_row if they exist
    if 'class_metrics' in metrics:
        # Only add per-class metrics to overall_row for single-label classification
        if not multilabel:
            for label, class_metrics in metrics['class_metrics'].items():
                prefix = f'class_{label}'
                overall_row.update({
                    f'{prefix}_roc_auc': class_metrics['roc_auc'],
                    f'{prefix}_avg_precision': class_metrics['avg_precision'],
                    f'{prefix}_precision': class_metrics['precision'],
                    f'{prefix}_recall': class_metrics['recall'],
                    f'{prefix}_f1': class_metrics['f1'],
                    f'{prefix}_accuracy': class_metrics['accuracy']
                })
        
    overall_row.update({
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'loss_fn': config['loss'],
        'loss': metrics['loss'],
        'timestamp': timestamp,
        'representation': config.get('representation', 'both'),
        'classification_head': config.get('classification_head', 'linear'),
        'seed': config.get('seed', "None")
        }
    )
    
    # Prepare data for per-class metrics CSV - only final values
    per_class_rows = []
    if 'class_metrics' in metrics:
        for label, class_metrics in metrics['class_metrics'].items():
            per_class_row = {
                'dataset': config['dataset'],
                'model': config['model'],
                'class': label,
                'roc_auc': class_metrics['roc_auc'],
                'avg_precision': class_metrics['avg_precision'],
                'timestamp': timestamp
            }
            per_class_rows.append(per_class_row)
    
    # Convert to dataframes
    overall_df = pd.DataFrame([overall_row])
    per_class_df = pd.DataFrame(per_class_rows) if per_class_rows else None
    
    # Save overall metrics - append if file exists
    if os.path.exists(overall_file):
        overall_df.to_csv(overall_file, mode='a', header=False, index=False)
    else:
        overall_df.to_csv(overall_file, index=False)
    
    # Save per-class metrics if they exist - append if file exists
    if per_class_df is not None:
        if os.path.exists(per_class_file):
            per_class_df.to_csv(per_class_file, mode='a', header=False, index=False)
        else:
            per_class_df.to_csv(per_class_file, index=False)
    
    return overall_file, per_class_file

def evaluate_model(model, data_loader, criterion, device, multilabel=True, label_columns=None, logger=None, config=None):
    """
    Evaluate model performance and return metrics.
    
    Args:
        model: The PyTorch model to evaluate
        data_loader: DataLoader containing test/validation data
        criterion: Loss function
        device: Device to run evaluation on
        multilabel: Whether this is multilabel classification
        label_columns: Names of the label columns for multilabel case
        logger: Logger instance for output
        config: Configuration dictionary
    """
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            # Convert labels to float only for multilabel case
            loss = criterion(outputs, labels.float() if multilabel else labels)
            total_loss += loss.item()
            
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all batches
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(data_loader)

    if multilabel:
        # Apply sigmoid for multilabel case
        all_outputs = torch.sigmoid(all_outputs)
        predictions = (all_outputs > 0.5).float()
        
        # Calculate metrics
        roc_auc = roc_auc_score(all_labels.numpy(), all_outputs.numpy(), average='macro')
        avg_precision = average_precision_score(all_labels.numpy(), all_outputs.numpy(), average='macro')
        f1 = f1_score(all_labels.numpy(), predictions.numpy(), average='macro')
        precision = precision_score(all_labels.numpy(), predictions.numpy(), average='macro')
        recall = recall_score(all_labels.numpy(), predictions.numpy(), average='macro')
        
        # Per-class metrics
        class_metrics = {}
        for i, col in enumerate(label_columns):
            class_metrics[col] = {
                'roc_auc': roc_auc_score(all_labels[:, i].numpy(), all_outputs[:, i].numpy()),
                'avg_precision': average_precision_score(all_labels[:, i].numpy(), all_outputs[:, i].numpy()),
                'f1': f1_score(all_labels[:, i].numpy(), predictions[:, i].numpy()),
                'precision': precision_score(all_labels[:, i].numpy(), predictions[:, i].numpy()),
                'recall': recall_score(all_labels[:, i].numpy(), predictions[:, i].numpy())
            }
        
        # Calculate category-wise metrics
        class_categories = categorize_classes(all_labels.numpy())
        category_metrics = {}
        
        for category, indices in class_categories.items():
            if indices:  # Only calculate if category has classes
                category_metrics[f'{category}_roc_auc'] = roc_auc_score(
                    all_labels[:, indices].numpy(), 
                    all_outputs[:, indices].numpy(), 
                    average='macro'
                )
                category_metrics[f'{category}_avg_precision'] = average_precision_score(
                    all_labels[:, indices].numpy(), 
                    all_outputs[:, indices].numpy(), 
                    average='macro'
                )
                
                if logger:
                    logger.info(f"\n{category.capitalize()} Class Metrics:")
                    logger.info(f"ROC-AUC: {category_metrics[f'{category}_roc_auc']:.4f}")
                    logger.info(f"Average Precision: {category_metrics[f'{category}_avg_precision']:.4f}")
                    logger.info(f"Classes: {[label_columns[i] for i in indices]}")

        # Add category metrics to the main metrics dictionary
        metrics = {
            'loss': avg_loss,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'class_metrics': class_metrics,
            'class_categories': {
                category: [label_columns[i] for i in indices] 
                for category, indices in class_categories.items()
            }
        }
        
        # Add category metrics to the main metrics dictionary
        metrics.update(category_metrics)
        
    else:
        # Apply softmax for single-label case
        probs = torch.softmax(all_outputs, dim=1)
        _, predictions = torch.max(all_outputs, 1)
        
        # Convert to numpy for sklearn metrics
        predictions = predictions.numpy()
        labels = all_labels.numpy()
        probs = probs.numpy()
        
        if logger:
            logger.info(f"\nNumber of classes: {probs.shape[1]}")
        
        # Calculate metrics
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='macro'),
            'precision': precision_score(labels, predictions, average='macro'),
            'recall': recall_score(labels, predictions, average='macro')
        }
        
        # Calculate per-class metrics
        metrics['class_metrics'] = {}
        for class_idx in range(probs.shape[1]):
            if logger:
                logger.info(f"\nCalculating metrics for class {class_idx}")
            
            # For multiclass, use one-vs-rest approach
            class_labels = (labels == class_idx)
            class_probs = probs[:, class_idx]
            class_preds = (predictions == class_idx)
            
            metrics['class_metrics'][str(class_idx)] = {
                'roc_auc': roc_auc_score(class_labels, class_probs),
                'avg_precision': average_precision_score(class_labels, class_probs),
                'precision': precision_score(class_labels, class_preds),
                'recall': recall_score(class_labels, class_preds),
                'f1': f1_score(class_labels, class_preds)
            }
            # Add accuracy only for single-label classification
            if not multilabel:
                metrics['class_metrics'][str(class_idx)]['accuracy'] = accuracy_score(class_labels, class_preds)
            
            if logger:
                logger.info(f"Class {class_idx} metrics:")
                for metric, value in metrics['class_metrics'][str(class_idx)].items():
                    logger.info(f"{metric}: {value:.4f}")
        
        # For binary classification
        if probs.shape[1] == 2:
            metrics.update({
                'roc_auc': roc_auc_score(labels, probs[:, 1]),
                'average_precision': average_precision_score(labels, probs[:, 1])
            })
        else:
            # For multiclass (including 3 classes)
            metrics.update({
                'roc_auc': roc_auc_score(labels, probs, average='macro', multi_class='ovr'),
                'average_precision': average_precision_score(labels, probs, average='macro')
            })
        
        if logger:
            logger.info("\nTest Metrics:")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"F1 Score: {metrics['f1']:.4f}")
            logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"Average Precision: {metrics['average_precision']:.4f}")
    
    return metrics

def track_metric_changes(metrics_history):
    """
    Analyze and print the changes in metrics every 5 epochs.
    
    Args:
        metrics_history: List of dictionaries containing metrics for each epoch
    """
    if not metrics_history:
        return
    
    # Get indices for every 5th epoch, plus the last epoch if not already included
    epoch_indices = list(range(0, len(metrics_history), 5))
    if (len(metrics_history) - 1) not in epoch_indices:
        epoch_indices.append(len(metrics_history) - 1)
    
    # Calculate changes for main metrics
    main_metrics = ['roc_auc', 'average_precision', 'precision', 'recall', 'f1']
    
    print("\n=== Metric Changes Over Training ===")
    
    # Print header with epoch numbers and change column
    header = "Metric".ljust(20)
    for idx in epoch_indices:
        header += f"Epoch {idx:>3}  "
    header += "Change    "
    print(header)
    print("-" * (20 + 10 * len(epoch_indices) + 10))
    
    # Print metrics for each epoch and change
    for metric in main_metrics:
        line = metric.ljust(20)
        for idx in epoch_indices:
            value = metrics_history[idx][metric]
            line += f"{value:>8.4f}  "
        
        # Add change column
        start_val = metrics_history[0][metric]
        end_val = metrics_history[-1][metric]
        change = end_val - start_val
        change_symbol = "↑" if change > 0 else "↓" if change < 0 else "="
        line += f"{change:>7.4f}{change_symbol}"
        print(line)
    
    # If multilabel, show per-class metric changes
    if 'class_metrics' in metrics_history[0]:
        print("\n=== Per-Class Metric Changes ===")
        for class_name in metrics_history[0]['class_metrics'].keys():
            print(f"\nClass: {class_name}")
            
            # Print header with epoch numbers and change column
            header = "Metric".ljust(20)
            for idx in epoch_indices:
                header += f"Epoch {idx:>3}  "
            header += "Change    "
            print(header)
            print("-" * (20 + 10 * len(epoch_indices) + 10))
            
            class_metrics = ['roc_auc', 'avg_precision', 'precision', 'recall', 'f1']
            for metric in class_metrics:
                line = metric.ljust(20)
                for idx in epoch_indices:
                    value = metrics_history[idx]['class_metrics'][class_name][metric]
                    line += f"{value:>8.4f}  "
                
                # Add change column
                start_val = metrics_history[0]['class_metrics'][class_name][metric]
                end_val = metrics_history[-1]['class_metrics'][class_name][metric]
                change = end_val - start_val
                change_symbol = "↑" if change > 0 else "↓" if change < 0 else "="
                line += f"{change:>7.4f}{change_symbol}"
                print(line)

    # Add category metrics tracking
    print("\n=== Category-wise Metric Changes ===")
    for category in ['head', 'medium', 'tail']:
        roc_key = f'{category}_roc_auc'
        ap_key = f'{category}_avg_precision'
        
        if roc_key in metrics_history[0]:
            print(f"\n{category.capitalize()} Classes:")
            if 'class_categories' in metrics_history[0]:
                print(f"Classes: {metrics_history[0]['class_categories'][category]}")
            
            for metric_key in [roc_key, ap_key]:
                line = metric_key.ljust(20)
                for idx in epoch_indices:
                    value = metrics_history[idx][metric_key]
                    line += f"{value:>8.4f}  "
                
                # Add change column
                start_val = metrics_history[0][metric_key]
                end_val = metrics_history[-1][metric_key]
                change = end_val - start_val
                change_symbol = "↑" if change > 0 else "↓" if change < 0 else "="
                line += f"{change:>7.4f}{change_symbol}"
                print(line)

def main():
    # Load config, setup logging, etc.
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    
    # Convert numeric config values to proper types
    config['learning_rate'] = float(config['learning_rate'])
    config['batch_size'] = int(config['batch_size'])
    config['num_epochs'] = int(config['num_epochs'])
    config['warmup_epochs'] = int(config['warmup_epochs'])
    if 'num_workers' in config:
        config['num_workers'] = int(config['num_workers'])
    if 'seed' in config:
        config['seed'] = int(config['seed'])
    
    logger = setup_logging(config)

    # Print config at start
    logger.info("\n" + "="*50)
    logger.info("Configuration:")
    logger.info("-"*50)
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    logger.info("="*50 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        if config['device'] == "mps":
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    logger.info(f"Using {device} for training")

    # Make wandb optional by checking if wandb-related configs exist
    use_wandb = config.get('use_wandb', False) and 'wandb_project' in config
    
    if use_wandb:
        init_wandb(config)
    
    # Get multilabel setting from dataset config
    dataset_config = get_dataset_config()[config['dataset']]
    config['multilabel'] = dataset_config.get('multilabel', config.get('multilabel', True))
    
    if not config['multilabel']:
        config['num_classes'] = dataset_config['num_classes']
    
    # Load data using the KNN scripts' loading logic
    train_embeddings = load_embeddings(config['base_path'], config['dataset'], 
                                     config['model'], 'train',
                                     representation=config.get('representation', 'both'))
    test_embeddings = load_embeddings(config['base_path'], config['dataset'], 
                                    config['model'], 'test',
                                    representation=config.get('representation', 'both'))
    
    if config['normalize_embeddings']:
        train_embeddings = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)

    # Update logging message to reflect representation type
    representation_type = config.get('representation', 'both')
    logger.info(f"Using {representation_type} representation")
    logger.info(f"Embedding shape: {train_embeddings.shape}")

    # Load labels
    train_labels, test_labels, label_columns = load_and_process_labels(
        config['base_path'], 
        config['dataset'],
        config['multilabel']
    )
    
    # Convert to tensors with proper dtype based on multilabel flag
    train_embeddings = torch.tensor(train_embeddings, dtype=torch.float32)
    test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32)
    
    # Use float32 for multilabel, long for single-label
    label_dtype = torch.float32 if config['multilabel'] else torch.long
    train_labels = torch.tensor(train_labels, dtype=label_dtype)
    test_labels = torch.tensor(test_labels, dtype=label_dtype)
    
    # Create dataloaders
    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    test_dataset = TensorDataset(test_embeddings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Setup model with specified head type
    head_type = config.get('classification_head', 'linear')
    model = LinearProbe(
        train_embeddings.shape[1],
        train_labels.shape[1] if config['multilabel'] else config['num_classes'],
        head_type=head_type
    ).to(device)
    
    # Print model architecture and parameters
    logger.info("\n" + "="*50)
    logger.info("Model Architecture:")
    logger.info("-"*50)
    logger.info(str(model))
    logger.info("="*50 + "\n")

    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info("-"*50 + "\n")

    # optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    optimizer = AdamW(model.parameters(), betas=(0.9, 0.95),
                      lr=float(config['learning_rate']), 
                      weight_decay=float(config['weight_decay']), decouple_lr=True)
    criterion = get_criterion(config, train_labels).to(device)
    
    # Setup scheduler
    num_training_steps = len(train_loader) * config['num_epochs']
    num_warmup_steps = len(train_loader) * config['warmup_epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    # Initialize metrics history
    metrics_history = []
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (embeddings, labels) in enumerate(train_loader):
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            # Convert labels to float only for multilabel case
            loss = criterion(outputs, labels.float() if config['multilabel'] else labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            if batch_idx % 5 == 0:
                logger.info(f"Step [{batch_idx}/{len(train_loader)}], "
                          f"LR: {scheduler.get_last_lr()[0]:.8f}, Loss: {loss.item():.4f}")
                if use_wandb:
                    log_training_step(loss, scheduler.get_last_lr()[0], epoch, 
                                    batch_idx, len(train_loader))
        
        # Evaluation
        model.eval()
        metrics = evaluate_model(
            model, test_loader, criterion, device, 
            config['multilabel'], label_columns,
            logger=logger,
            config=config
        )
        
        # Store metrics for tracking
        metrics_history.append(metrics)
        
        # Logging
        avg_loss = running_loss / len(train_loader)
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch [{epoch+1}/{config['num_epochs']}], Train Loss: {avg_loss:.4f}")
        logger.info(f"Test Metrics Summary:")
        logger.info("-"*50)
        logger.info(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
        logger.info(f"Average Precision:   {metrics['average_precision']:.4f}")
        logger.info(f"Precision:          {metrics['precision']:.4f}")
        logger.info(f"Recall:             {metrics['recall']:.4f}")
        logger.info(f"F1 Score:           {metrics['f1']:.4f}")

        if config['multilabel']:
            logger.info("\nPer-class Performance:")
            logger.info("-"*50)
            logger.info(f"{'Class':<25} {'ROC-AUC':>10} {'AP':>10} {'Prec':>10} {'Recall':>10} {'F1':>10}")
            logger.info("-"*75)
            
            for label, class_metric in metrics['class_metrics'].items():
                logger.info(f"{label:<25} "
                          f"{class_metric['roc_auc']:>10.4f} "
                          f"{class_metric['avg_precision']:>10.4f} "
                          f"{class_metric['precision']:>10.4f} "
                          f"{class_metric['recall']:>10.4f} "
                          f"{class_metric['f1']:>10.4f}")
            logger.info("="*50 + "\n")

        if use_wandb:
            log_epoch_metrics(avg_loss, None, metrics, epoch, config['multilabel'])
            
            # Log test results on final epoch
            if epoch == config['num_epochs'] - 1:
                log_test_results(config, metrics['results_df'], epoch)
    
    # Print metric changes at the end of training
    track_metric_changes(metrics_history)
    
    # Save only the final metrics to CSV
    save_results_to_csv(metrics_history[-1], config)
    
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 