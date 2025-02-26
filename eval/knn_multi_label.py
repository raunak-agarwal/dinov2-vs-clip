"""
Multi-label kNN Classification for Medical Image Embeddings

This script performs kNN classification on medical image embeddings for multiple datasets
and models. It supports both CLIP and DINO embeddings across various medical datasets.

Usage:
    python knn_multi_label.py [options]

Options:
    --base_path      Base directory containing embeddings and label files
                     Default: "../process_outputs/embeddings-with-labels"
    
    --datasets       Space-separated list of datasets to evaluate
                     Choices: mimic, mimic_lt, chexpert, nih, nih_lt, vindr-cxr, vindr-pcxr, all
                     Default: all
    
    --models         Space-separated list of models to evaluate
                     Choices: clip, dino, all
                     Default: all
    
    --k_values       Space-separated list of k values for kNN classifier
                     Special values: 
                       - "default": [9, 19, 29, 43, 59, 71, 89, 100]
                       - "all": all odd numbers from 1 to 100
                     Or specify custom values: e.g., "5 10 15 20"
                     Default: default
    
    --n_jobs        Number of parallel jobs for kNN classifier
                    Default: 8
    
    --output_dir    Directory to save results
                    Default: "results"

Examples:
    # Run with default settings (all datasets, all models, default k values)
    python knn_multi_label.py

    # Run specific dataset with specific model
    python knn_multi_label.py --datasets mimic --models clip

    # Run multiple datasets with custom k values
    python knn_multi_label.py --datasets mimic chexpert --k_values 5 10 15 20

    # Run comprehensive k value search
    python knn_multi_label.py --datasets mimic --models clip --k_values all

    # Run with custom parallel jobs and output directory
    python knn_multi_label.py --n_jobs 8 --output_dir custom_results
"""

import numpy as np
import pandas as pd
from skmultilearn.adapt import MLkNN, BRkNNbClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import json
from datetime import datetime
import os
import argparse

def load_embeddings(base_path, dataset_name, model_name, split):
    """Load embeddings for a specific dataset, model and split"""
    # Handle special case for vindr dataset names in file paths
    file_dataset_name = dataset_name.replace("-", "-")  # Keep as is for normal cases
    if dataset_name in ['vindr-cxr', 'vindr-pcxr']:
        file_dataset_name = dataset_name.replace("-", "-")
    if dataset_name in ["mimic_lt", "nih_lt"]:
        dataset_name = dataset_name.replace("_", "-")
    
    # Use validation split instead of test for CheXpert
    if dataset_name == "chexpert" and split == "test":
        split = "val"
    
    path = f"{base_path}/{dataset_name}-embeddings/{file_dataset_name}_{model_name}_{split}_embeddings.npy"
    return np.load(path)

def load_and_process_labels(labels_path, columns, dataset_name):
    """Load and process labels dataframe"""
    labels = pd.read_csv(labels_path, sep='\t')
    
    # For CheXpert, use validation split instead of test
    if dataset_name == "chexpert":
        train_labels = labels[labels['split'] == 'train']
        test_labels = labels[labels['split'] == 'valid']
    else:
        train_labels = labels[labels['split'] == 'train']
        test_labels = labels[labels['split'] == 'test']
    
    # Select only the disease columns
    train_labels = train_labels[columns]
    test_labels = test_labels[columns]
    
    # Replace -1 with 1 and NaN with 0
    train_labels = train_labels.replace(-1, 1).fillna(0)
    test_labels = test_labels.replace(-1, 1).fillna(0)
    
    return train_labels.to_numpy(), test_labels.to_numpy()

def evaluate_model(y_true, y_scores, class_names):
    """Evaluate model performance and return metrics"""
    results = {}
    
    results['overall'] = {
        'roc_auc': roc_auc_score(y_true, y_scores),
        'average_precision': average_precision_score(y_true, y_scores)
    }
    results['per_class'] = {}
    for i, class_name in enumerate(class_names):
        results['per_class'][class_name] = {
            'roc_auc': roc_auc_score(y_true[:, i], y_scores[:, i]),
            'average_precision': average_precision_score(y_true[:, i], y_scores[:, i])
        }
    
    return results

def save_results(results, dataset_name, model_name, output_dir="results"):
    """Save evaluation results to a JSON file"""
    # Create dataset-specific directory
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    filename = f"{dataset_dir}/{model_name}_results.json"
    
    # Load existing results if file exists
    existing_results = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_results = json.load(f)
    
    # Update with new results
    existing_results.update(results)
    
    with open(filename, 'w') as f:
        json.dump(existing_results, f, indent=4)
    
    return filename

def save_results_to_csv(results, dataset_name, model_name, output_dir="results"):
    """Save results to CSV files - one for overall metrics and one for per-class metrics"""
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Prepare data for overall metrics CSV
    overall_data = []
    per_class_data = []
    
    config = results['config']
    for k, k_results in results['results'].items():
        # Overall metrics
        overall_row = {
            'dataset': dataset_name,
            'model': model_name,
            'k': k,
            'roc_auc': k_results['overall_roc_auc'],
            'average_precision': k_results['overall_average_precision'],
            'time_taken_seconds': k_results['time_taken_seconds'],
            'timestamp': config['timestamp'],
            'n_jobs': config['n_jobs']
        }
        overall_data.append(overall_row)
        
        # Per-class metrics
        for class_name, class_metrics in k_results['per_class'].items():
            per_class_row = {
                'dataset': dataset_name,
                'model': model_name,
                'k': k,
                'class': class_name,
                'roc_auc': class_metrics['roc_auc'],
                'average_precision': class_metrics['average_precision'],
                'timestamp': config['timestamp']
            }
            per_class_data.append(per_class_row)
    
    # Save to CSV files
    overall_df = pd.DataFrame(overall_data)
    per_class_df = pd.DataFrame(per_class_data)
    
    overall_file = f"{dataset_dir}/{model_name}_overall_metrics.csv"
    per_class_file = f"{dataset_dir}/{model_name}_per_class_metrics.csv"
    
    # If files exist, append without header
    if os.path.exists(overall_file):
        overall_df.to_csv(overall_file, mode='a', header=False, index=False)
    else:
        overall_df.to_csv(overall_file, index=False)
        
    if os.path.exists(per_class_file):
        per_class_df.to_csv(per_class_file, mode='a', header=False, index=False)
    else:
        per_class_df.to_csv(per_class_file, index=False)
    
    return overall_file, per_class_file

def get_dataset_config():
    """Get configuration for datasets"""
    return {
        "mimic": {
            "label_file": "mimic-cxr-2.0.0-merged-with-paths.csv",
            "columns": ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                       'Lung Opacity', 'No Finding', 'Pleural Effusion',
                       'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
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
    }

def print_dataset_stats(train_labels, test_labels, columns):
    """Print dataset statistics"""
    print("\nDataset Statistics:")
    print(f"Training set size: {train_labels.shape[0]}")
    print(f"Test set size: {test_labels.shape[0]}")
    print(f"Number of labels: {len(columns)}")
    
    print("\nLabel distribution:")
    for i, col in enumerate(columns):
        train_pos = np.sum(train_labels[:, i] == 1)
        train_pct = (train_pos / len(train_labels)) * 100
        test_pos = np.sum(test_labels[:, i] == 1)
        test_pct = (test_pos / len(test_labels)) * 100
        
        print(f"{col}:")
        print(f"  Train: {train_pos}/{len(train_labels)} ({train_pct:.1f}%)")
        print(f"  Test:  {test_pos}/{len(test_labels)} ({test_pct:.1f}%)")

def run_linear_probing_multiple_k(base_path, dataset_name, model_name, k_values, n_jobs=8):
    """Run linear probing evaluation for multiple k values"""
    start_time = datetime.now()
    
    dataset_configs = get_dataset_config()
    config = dataset_configs[dataset_name]
    
    # Load embeddings
    print(f"Loading embeddings for {dataset_name}...")
    train_embeddings = load_embeddings(base_path, dataset_name, model_name, 'train')
    test_embeddings = load_embeddings(base_path, dataset_name, model_name, 'test')
    
    # Load and process labels with dataset_name parameter
    print("Loading and processing labels...")
    labels_path = os.path.join(base_path, config['label_file'])
    train_labels, test_labels = load_and_process_labels(
        labels_path, 
        config['columns'],
        dataset_name
    )
    
    # Print dataset statistics
    print_dataset_stats(train_labels, test_labels, config['columns'])
    
    all_results = {}
    k_results = {}
    
    for k in k_values:
        k_start_time = datetime.now()
        print(f"\nTraining classifier with k={k}...")
        
        try:
            # Train classifier
            # classifier = MLkNN(k=k, n_jobs=n_jobs)
            classifier = 
            # Convert labels to int type and ensure they're dense arrays
            classifier.fit(train_embeddings, train_labels.astype(int))
            
            # Get predictions
            scores = classifier.predict_proba(test_embeddings)
            # Convert sparse matrix to dense array
            scores = scores.toarray()
            
            # Evaluate and get results
            results = evaluate_model(test_labels, scores, config['columns'])
            
            k_time_taken = datetime.now() - k_start_time
            
            # Store results for this k
            k_results[str(k)] = {
                'overall_roc_auc': results['overall']['roc_auc'],
                'overall_average_precision': results['overall']['average_precision'],
                'per_class': results['per_class'],
                'time_taken_seconds': k_time_taken.total_seconds()
            }
            
            # Print current k results
            print(f"k={k} - ROC-AUC: {results['overall']['roc_auc']:.3f}, "
                  f"AP: {results['overall']['average_precision']:.3f}")
            print(f"Time taken for k={k}: {k_time_taken}")
            
        except Exception as e:
            print(f"Error processing k={k}: {str(e)}")
            continue
    
    total_time = datetime.now() - start_time
    
    # Prepare final results
    all_results = {
        'config': {
            'dataset_name': dataset_name,
            'model_name': model_name,
            'n_jobs': n_jobs,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'total_time_seconds': total_time.total_seconds()
        },
        'results': k_results
    }
    
    # Save all results
    output_file = save_results(all_results, dataset_name, model_name)
    print(f"\nResults saved to: {output_file}")
    
    # Save results to CSV
    overall_csv, per_class_csv = save_results_to_csv(all_results, dataset_name, model_name)
    print(f"Overall metrics saved to: {overall_csv}")
    print(f"Per-class metrics saved to: {per_class_csv}")
    
    print(f"Total time taken: {total_time}")
    
    return k_results, output_file

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run kNN multi-label classification on medical datasets')
    
    parser.add_argument('--base_path', type=str, 
                       default="../process_outputs/embeddings-with-labels",
                       help='Base path for embeddings and label files')
    
    parser.add_argument('--datasets', nargs='+',
                       choices=["mimic", "mimic_lt", "chexpert", "nih", "nih_lt", 
                               "vindr-cxr", "vindr-pcxr", "all"],
                       default=["all"],
                       help='Datasets to evaluate. Use "all" for all datasets')
    
    parser.add_argument('--models', nargs='+',
                       choices=['clip', 'dino', 'all'],
                       default=["all"],
                       help='Models to evaluate. Use "all" for all models')
    
    parser.add_argument('--k_values', nargs='+', type=str,
                       default=["default"],
                       help='k values for kNN classifier. Use "all" for comprehensive range, '
                            '"default" for preset values, or specify space-separated integers')
    
    parser.add_argument('--n_jobs', type=int, default=8,
                       help='Number of parallel jobs for kNN classifier')
    
    parser.add_argument('--output_dir', type=str, default="results",
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Process k_values argument
    if "all" in args.k_values:
        # Comprehensive range of k values
        args.k_values = list(range(1, 101, 2))  # Odd numbers from 1 to 100
    elif "default" in args.k_values:
        # Default preset values
        args.k_values = [3, 7, 11, 15, 23, 29]
    else:
        # Convert string inputs to integers
        try:
            args.k_values = [int(k) for k in args.k_values]
        except ValueError:
            raise ValueError("k_values must be integers, 'all', or 'default'")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # Define all available datasets and models
    all_datasets = ["mimic", "mimic_lt", "chexpert", "nih", "nih_lt", "vindr-cxr"]
    all_models = ['clip', 'dino']
    
    # Determine which datasets to process
    datasets_to_process = all_datasets if "all" in args.datasets else args.datasets
    models_to_process = all_models if "all" in args.models else args.models
    
    print("\nRunning with configuration:")
    print(f"Datasets: {datasets_to_process}")
    print(f"Models: {models_to_process}")
    print(f"k values: {args.k_values}")
    print(f"Number of jobs: {args.n_jobs}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Create main results directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Track overall progress
    total_combinations = len(datasets_to_process) * len(models_to_process)
    current_combination = 0
    
    for dataset_name in datasets_to_process:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        for model_name in models_to_process:
            current_combination += 1
            print(f"\nEvaluating {model_name} model... ({current_combination}/{total_combinations})")
            
            try:
                k_results, summary_file = run_linear_probing_multiple_k(
                    args.base_path, 
                    dataset_name, 
                    model_name,
                    args.k_values,
                    args.n_jobs
                )
                
                # Print summary of best k
                best_k = max(k_results.items(), key=lambda x: x[1]['overall_roc_auc'])
                print(f"\nBest k for {dataset_name} with {model_name}:")
                print(f"k={best_k[0]} - ROC-AUC: {best_k[1]['overall_roc_auc']:.3f}, "
                      f"AP: {best_k[1]['overall_average_precision']:.3f}")
                
            except Exception as e:
                print(f"Error processing {dataset_name} with {model_name}: {str(e)}")
                continue



