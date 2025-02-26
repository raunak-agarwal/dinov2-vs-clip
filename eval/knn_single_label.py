"""
Single-label kNN Classification for Medical Image Embeddings

This script performs kNN classification on medical image embeddings for multiple datasets
and models. It supports both CLIP and DINO embeddings across various medical datasets.

Usage:
    python knn_single_label.py [options]

Options:
    --base_path      Base directory containing embeddings and label files
                     Default: "../process_outputs/embeddings-with-labels"
    
    --datasets       Space-separated list of datasets to evaluate
                     Choices: covid, chexchonet, tbx11k, rsna, siim, all
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
                    Default: "results_single_label"

Examples:
    # Run with default settings (all datasets, all models, default k values)
    python knn_single_label.py

    # Run specific dataset with specific model
    python knn_single_label.py --datasets covid --models clip

    # Run multiple datasets with custom k values
    python knn_single_label.py --datasets covid rsna --k_values 5 10 15 20

    # Run comprehensive k value search
    python knn_single_label.py --datasets covid --models clip --k_values all

    # Run with custom parallel jobs and output directory
    python knn_single_label.py --n_jobs 8 --output_dir custom_results
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import json
from datetime import datetime
import os
import argparse

def get_embeddings_dir(dataset_name):
    """Get the correct embeddings directory name for a dataset"""
    dir_mapping = {
        "rsna": "rsna-pneumonia-embeddings",
        "siim": "siim-pneumothorax-embeddings",
        "tbx11k": "tbx11k-embeddings",
        "covid": "covid-embeddings",
        "chexchonet": "chexchonet-embeddings"
    }
    return dir_mapping.get(dataset_name, f"{dataset_name}-embeddings")

def load_embeddings(base_path, dataset_name, model_name, split):
    """Load embeddings for a specific dataset, model and split"""
    embeddings_dir = get_embeddings_dir(dataset_name)
    path = f"{base_path}/{embeddings_dir}/{dataset_name}_{model_name}_{split}_embeddings.npy"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Embedding file not found: {path}\n"
            f"Please ensure the embeddings for {dataset_name} {model_name} model are generated first."
        )
    return np.load(path)

def load_and_process_labels(labels_path):
    """Load and process labels dataframe"""
    labels = pd.read_csv(labels_path, sep='\t')
    
    # Split into train and test
    train_labels = labels[labels['split'] == 'train']['label']
    test_labels = labels[labels['split'] == 'test']['label']
    
    # Convert to numpy arrays
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    # Ensure both are 1D arrays
    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()
    
    # Convert to integers
    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)
    
    return train_labels, test_labels

def evaluate_model(y_true, y_scores, y_pred):
    """Evaluate model performance and return metrics"""
    # print(y_true.shape, y_scores.shape, y_pred.shape)
    # print(y_scores)
    
    # Get classification report metrics
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate ROC AUC based on number of classes
    n_classes = len(set(y_true))
    if n_classes == 2:
        # Binary classification - use probability of positive class (class 1)
        # y_scores[:, 1] gives us the probability for class 1
        roc_auc = roc_auc_score(y_true, y_scores[:, 1])
        # For average precision in binary case, also use probability of positive class
        avg_precision = average_precision_score(y_true, y_scores[:, 1])
    else:
        # Multi-class - use OvR approach
        roc_auc = roc_auc_score(y_true, y_scores, multi_class='ovr')
        # For multi-class, keep using full probability matrix
        avg_precision = average_precision_score(y_true, y_scores, average='macro')
    
    # Extract overall metrics from classification report
    results = {
        'overall': {
            'accuracy': class_report['accuracy'],
            'macro_precision': class_report['macro avg']['precision'],
            'macro_recall': class_report['macro avg']['recall'],
            'macro_f1': class_report['macro avg']['f1-score'],
            'roc_auc': roc_auc,
            'average_precision': avg_precision
        },
        'classification_report': class_report
    }
    return results

def save_results(results, dataset_name, model_name, output_dir="results_single_label"):
    """Save evaluation results to a JSON file"""
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

def save_summary(dataset_name, model_name, k_results, output_dir="results_single_label"):
    """Save summary of results for all k values"""
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_dir}/{model_name}_summary_{timestamp}.json"
    
    summary = {
        'dataset': dataset_name,
        'model': model_name,
        'k_values': k_results
    }
    
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=4)
    
    return filename

def get_dataset_config():
    """Get configuration for single-label datasets"""
    return {
        "covid": {
            "label_file": "covid-merged.csv"
        },
        "chexchonet": {
            "label_file": "chexchonet-merged.csv"
        },
        "tbx11k": {
            "label_file": "tbx11k-merged.csv"
        },
        "rsna": {
            "label_file": "rsna-pneumonia-merged.csv"
        },
        "siim": {
            "label_file": "siim-pneumothorax-merged.csv"
        }
    }

def print_dataset_stats(train_labels, test_labels):
    """Print dataset statistics"""
    print("\nDataset Statistics:")
    print(f"Training set size: {len(train_labels)}")
    print(f"Test set size: {len(test_labels)}")
    
    print("\nLabel distribution:")
    unique_labels = np.unique(np.concatenate([train_labels, test_labels]))
    for label in unique_labels:
        train_count = np.sum(train_labels == label)
        train_pct = (train_count / len(train_labels)) * 100
        test_count = np.sum(test_labels == label)
        test_pct = (test_count / len(test_labels)) * 100
        
        print(f"Class {label}:")
        print(f"  Train: {train_count}/{len(train_labels)} ({train_pct:.1f}%)")
        print(f"  Test:  {test_count}/{len(test_labels)} ({test_pct:.1f}%)")

def run_knn_multiple_k(base_path, dataset_name, model_name, k_values, n_jobs=8):
    """Run KNN evaluation for multiple k values"""
    start_time = datetime.now()
    
    dataset_configs = get_dataset_config()
    config = dataset_configs[dataset_name]
    
    # Load embeddings
    print(f"Loading embeddings for {dataset_name}...")
    train_embeddings = load_embeddings(base_path, dataset_name, model_name, 'train')
    test_embeddings = load_embeddings(base_path, dataset_name, model_name, 'test')
    
    # Load and process labels
    print("Loading and processing labels...")
    labels_path = os.path.join(base_path, config['label_file'])
    train_labels, test_labels = load_and_process_labels(labels_path)   

    # Print dataset statistics
    print_dataset_stats(train_labels, test_labels)
    
    all_results = {}
    k_results = {}
    
    for k in k_values:
        k_start_time = datetime.now()
        print(f"\nTraining classifier with k={k}...")
        
        # Train classifier
        classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs)
        classifier.fit(train_embeddings, train_labels)
        print("Done training")
        
        # Get predictions and probabilities
        y_pred = classifier.predict(test_embeddings)
        y_scores = classifier.predict_proba(test_embeddings)
        print("Done predicting scores")
        
        # Evaluate and get results
        results = evaluate_model(test_labels, y_scores, y_pred)
        
        k_time_taken = datetime.now() - k_start_time
        
        # Store results for this k
        k_results[str(k)] = {
            'overall_accuracy': results['overall']['accuracy'],
            'overall_macro_precision': results['overall']['macro_precision'],
            'overall_macro_recall': results['overall']['macro_recall'],
            'overall_macro_f1': results['overall']['macro_f1'],
            'overall_roc_auc': results['overall']['roc_auc'],
            'overall_average_precision': results['overall']['average_precision'],
            'classification_report': results['classification_report'],
            'time_taken_seconds': k_time_taken.total_seconds()
        }
        
        # Print current k results
        print(f"k={k}:")
        print(f"  Accuracy: {results['overall']['accuracy']:.3f}")
        print(f"  Macro metrics:")
        print(f"    Precision: {results['overall']['macro_precision']:.3f}")
        print(f"    Recall: {results['overall']['macro_recall']:.3f}")
        print(f"    F1: {results['overall']['macro_f1']:.3f}")
        print(f"  ROC-AUC: {results['overall']['roc_auc']:.3f}")
        print(f"  AP: {results['overall']['average_precision']:.3f}")
        print(f"Time taken for k={k}: {k_time_taken}")
    
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
    parser = argparse.ArgumentParser(description='Run kNN single-label classification on medical datasets')
    
    parser.add_argument('--base_path', type=str, 
                       default="../process_outputs/embeddings-with-labels",
                       help='Base path for embeddings and label files')
    
    parser.add_argument('--datasets', nargs='+',
                       choices=["covid", "chexchonet", "tbx11k", "rsna", "siim", "all"],
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
    
    parser.add_argument('--output_dir', type=str, default="results_single_label",
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

def save_results_to_csv(results, dataset_name, model_name, output_dir="results_single_label"):
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
            'accuracy': k_results['overall_accuracy'],
            'macro_precision': k_results['overall_macro_precision'],
            'macro_recall': k_results['overall_macro_recall'],
            'macro_f1': k_results['overall_macro_f1'],
            'roc_auc': k_results['overall_roc_auc'],
            'average_precision': k_results['overall_average_precision'],
            'time_taken_seconds': k_results['time_taken_seconds'],
            'timestamp': config['timestamp'],
            'n_jobs': config['n_jobs']
        }
        overall_data.append(overall_row)
        
        # Per-class metrics from classification report
        for class_label, metrics in k_results['classification_report'].items():
            if class_label in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            per_class_row = {
                'dataset': dataset_name,
                'model': model_name,
                'k': k,
                'class': class_label,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1-score'],
                'support': metrics['support'],
                'timestamp': config['timestamp']
            }
            per_class_data.append(per_class_row)
    
    # Save to CSV files - using dataset name only (not model specific)
    overall_df = pd.DataFrame(overall_data)
    per_class_df = pd.DataFrame(per_class_data)
    
    overall_file = f"{dataset_dir}/overall_metrics.csv"
    per_class_file = f"{dataset_dir}/per_class_metrics.csv"
    
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

if __name__ == "__main__":
    args = parse_args()
    
    # Define all available datasets and models
    all_datasets = ["covid", "chexchonet", "tbx11k", "rsna", "siim"]
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
                k_results, summary_file = run_knn_multiple_k(
                    args.base_path, 
                    dataset_name, 
                    model_name,
                    args.k_values,
                    args.n_jobs
                )
                
                # Print summary of best k (using macro F1 score as the metric)
                best_k = max(k_results.items(), key=lambda x: x[1]['overall_macro_f1'])
                print(f"\nBest k for {dataset_name} with {model_name}:")
                print(f"k={best_k[0]}:")
                print(f"  Accuracy: {best_k[1]['overall_accuracy']:.3f}")
                print(f"  Macro metrics:")
                print(f"    Precision: {best_k[1]['overall_macro_precision']:.3f}")
                print(f"    Recall: {best_k[1]['overall_macro_recall']:.3f}")
                print(f"    F1: {best_k[1]['overall_macro_f1']:.3f}")
                print(f"  ROC-AUC: {best_k[1]['overall_roc_auc']:.3f}")
                print(f"  AP: {best_k[1]['overall_average_precision']:.3f}")
                
            except Exception as e:
                print(f"Error processing {dataset_name} with {model_name}: {str(e)}")
                continue 