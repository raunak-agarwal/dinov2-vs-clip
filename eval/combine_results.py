import os
import pandas as pd
from pathlib import Path
from datetime import datetime

def get_metrics_filename(dataset):
    """
    Returns the appropriate metrics filename based on the dataset.
    """
    # Map dataset names to their specific filenames
    dataset_file_map = {
        'covid': 'overall_metrics_covid.csv',
        'rsna-pneumonia': 'overall_metrics_rsna-pneumonia.csv',
        'chexchonet': 'overall_metrics_chexchonet.csv',
        'siim-pneumothorax': 'overall_metrics_siim-pneumothorax.csv',
        'tbx11k': 'overall_metrics_tbx11k.csv'
    }
    
    # For multilabel datasets
    multilabel_datasets = ['mimic', 'nih', 'chexpert', 'vindr-cxr', 'mimic_lt', 'nih_lt']
    
    if dataset in multilabel_datasets:
        return "overall_metrics_multilabel.csv"
    else:
        return dataset_file_map.get(dataset, "overall_metrics.csv")

def combine_results():
    """
    Combines metrics files from subdirectories in the runs folder into separate
    combined files based on dataset type.
    """
    # Get the runs directory
    runs_dir = Path("runs")
    
    if not runs_dir.exists():
        print("No 'runs' directory found.")
        return
    
    # Initialize dictionaries to store dataframes by dataset type
    dataset_dfs = {
        'multilabel': [],
        'covid': [],
        'rsna-pneumonia': [],
        'chexchonet': [],
        'siim-pneumothorax': [],
        'tbx11k': []
    }
    
    # Walk through all subdirectories in runs folder
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Try to determine the dataset from the metrics file content
        for metrics_file in run_dir.glob("overall_metrics*.csv"):
            try:
                # Read the first row to get the dataset name
                df = pd.read_csv(metrics_file)
                if len(df) > 0:
                    dataset = df.iloc[0]['dataset']
                    expected_filename = get_metrics_filename(dataset)
                    
                    # Check if this is the correct file for this dataset
                    if metrics_file.name == expected_filename:
                        df['run_name'] = run_dir.name
                        
                        # Add to appropriate dataset group
                        if metrics_file.name == 'overall_metrics_multilabel.csv':
                            dataset_dfs['multilabel'].append(df)
                        elif dataset in dataset_dfs:
                            dataset_dfs[dataset].append(df)
                            
                        print(f"Loaded metrics from: {run_dir.name} ({metrics_file.name})")
                    else:
                        print(f"Skipping {metrics_file.name} as it doesn't match expected filename {expected_filename} for dataset {dataset}")
            except Exception as e:
                print(f"Error reading {metrics_file}: {e}")
    
    # Combine and save results for each dataset type
    for dataset_type, dfs in dataset_dfs.items():
        if dfs:  # Only process if we have data for this dataset type
            # Combine dataframes for this dataset type
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Sort by model and timestamp
            combined_df = combined_df.sort_values(['model', 'timestamp'])
            
            # Create output filename based on dataset type
            if dataset_type == 'multilabel':
                output_file = runs_dir / "overall_metrics_multilabel.csv"
            else:
                output_file = runs_dir / f"overall_metrics_{dataset_type}.csv"
            
            # Save combined results
            combined_df.to_csv(output_file, index=False)
            print(f"\nCreated {output_file} with {len(combined_df)} entries")

if __name__ == "__main__":
    combine_results() 