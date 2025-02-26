# python create_embeddings.py --cfg embeddings_vindrcxr.yaml

""" 
Settings can be set in a yaml file as follows:

# For DINO models only
dino_config: "dino_vitb16_config.yaml"
dino_model_path: "training_99999.pth"

# For CLIP models only
clip_config: "path/to/clip-config.json"  # Required if using CLIP
clip_model_path: "cxr-bert-epoch_50.pt" # Required if using CLIP
clip_model_name: "cxrclip_local"  # Required if using CLIP

# Dataset settings
finetune_dataset: "vindr-cxr"  # Dataset name that matches one in dataloaders.py
image_dir: "path/to/images"
label_file: "path/to/labels.csv"

# Optional dataset settings
drop_columns: "No finding"  # Columns to drop, separate multiple with ";"
sampling_ratio: null  # Optional: fraction of data to sample
min_samples_per_label: null  # Optional: minimum samples per label when sampling

# Processing settings
batch_size: 32  # Batch size for data loading
num_workers: 12  # Number of workers for data loading
output_dir: "embeddings"  # Directory to save embeddings
dino_or_clip: "both"  # "both", "dino", "clip"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import json
import argparse
from pathlib import Path
import numpy as np
import os
from improved_dataloaders import get_dataset
from modeling import DINOEmbeddingModel
from open_clip import create_model_and_transforms
from open_clip.factory import _MODEL_CONFIGS
from datetime import datetime

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True,
                      help='Path to config YAML file')
    return parser.parse_args()

def get_dino_model(model_args: dict, training_args: dict):
    # Update training_args with the model path key as DINOEmbeddingModel expects it
    training_args['model_path'] = training_args['dino_model_path']
    model = DINOEmbeddingModel(model_args, training_args)
    return model, None  # Return None for preprocess to match CLIP interface

def get_clip_model(model_args: dict, training_args: dict):
    # Add validation
    if not os.path.exists(training_args['clip_config']):
        raise FileNotFoundError(f"CLIP config not found at {training_args['clip_config']}")
    if not os.path.exists(training_args['clip_model_path']):
        raise FileNotFoundError(f"CLIP model not found at {training_args['clip_model_path']}")
        
    # Load CLIP config
    with open(training_args['clip_config'], 'r') as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]
    
    model_name = training_args['clip_model_name']
    _MODEL_CONFIGS[model_name] = model_cfg
    
    try:
        model, _, preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained=training_args['clip_model_path'],
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create CLIP model: {str(e)}")
    
    return model, preprocess

def compute_embeddings(dataloader, model, preprocess, device, config):
    """Compute embeddings for a dataset"""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            # Extract images from batch - batch should be (images, labels)
            images = batch[0]  # First element is always images
            
            # Handle preprocessing
            if preprocess:  # For CLIP
                # Process each image in the batch
                processed_images = []
                for img in images:
                    processed_images.append(preprocess(img))
                images = torch.stack(processed_images)
            else:  # For DINO
                # Images should already be a tensor from the collate_fn
                if not isinstance(images, torch.Tensor):
                    raise TypeError(f"Expected tensor for DINO but got {type(images)}")
                    
            # Move to device
            images = images.to(device)
            
            # Get features
            if isinstance(model, DINOEmbeddingModel):
                features = model(images, get_intermediate_layers=config.get('get_intermediate_layers', False))
            else:  # CLIP model
                if not config.get('get_intermediate_layers', False):
                    features = model.encode_image(images)
                else:
                    features1 = model.encode_image(images) # 1, 768
                    features2 = model.visual.trunk.forward_intermediates(images, indices=1, output_fmt="NLC", norm=True, intermediates_only=True)[0].mean(dim=1) # 1, 768
                    features = torch.cat([features1, features2], dim=1) # 1, 1536
                
            all_embeddings.append(features.cpu())
            
    return torch.cat(all_embeddings, dim=0)

def get_embeddings_filename(model_type: str, split: str, config: dict) -> str:
    """Generate filename for embeddings based on config and split"""
    dataset = config['finetune_dataset']
    return f"{dataset}_{model_type}_{split}_embeddings.npy"

def process_embeddings(model, preprocess, dataloaders, device, model_type: str, config: dict, output_dir: Path):
    """Process embeddings for a specific model type"""
    train_loader, val_loader, test_loader = dataloaders
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Computing {model_type} train embeddings...")
    train_embeddings = compute_embeddings(train_loader, model, preprocess, device, config)
    train_filename = get_embeddings_filename(model_type, 'train', config)
    np.save(os.path.join(output_dir, train_filename), train_embeddings.numpy())
    
    if val_loader:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Computing {model_type} validation embeddings...")
        val_embeddings = compute_embeddings(val_loader, model, preprocess, device, config)
        val_filename = get_embeddings_filename(model_type, 'val', config)
        np.save(os.path.join(output_dir, val_filename), val_embeddings.numpy())
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Computing {model_type} test embeddings...")
    test_embeddings = compute_embeddings(test_loader, model, preprocess, device, config)
    test_filename = get_embeddings_filename(model_type, 'test', config)
    np.save(os.path.join(output_dir, test_filename), test_embeddings.numpy())
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {model_type} embeddings saved to {output_dir}")
    print(f"Train embeddings shape: {train_embeddings.shape} -> {train_filename}")
    if val_loader:
        print(f"Validation embeddings shape: {val_embeddings.shape} -> {val_filename}")
    print(f"Test embeddings shape: {test_embeddings.shape} -> {test_filename}")
    
def dataloaders_from_config(config: dict, no_transform: bool = False):
    """Create dataloaders with custom collate function for CLIP if needed"""
    config['no_transform'] = no_transform
    train_dataset, val_dataset, test_dataset = get_dataset(config)
    
    # Define a custom collate function that doesn't try to stack PIL images
    def collate_fn(batch):
        # Separate images and labels
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        if no_transform:
            # For CLIP: return list of PIL images
            return images, torch.stack(labels)
        else:
            # For DINO: stack tensors
            return torch.stack(images), torch.stack(labels)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.get('batch_size', 128),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 128),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 128),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader, test_loader

def main():
    args = arg_parser()
    
    # Load config
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
    
    # Explicitly remove seed from config to prevent shuffling in datasets
    config.pop('seed', None)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Using device: {device}")
    
    # Create output directory if specified
    output_dir = Path(config.get('output_dir', '.'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model type from config, default to "both"
    model_type = config.get('dino_or_clip', 'both').lower()
    
    if model_type in ['both', 'dino']:
        # Get dataloaders for DINO with preprocessing
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Loading DataLoaders for DINO. no_transform=False...")
        dataloaders = dataloaders_from_config(config, no_transform=False)
        
        # Process DINO embeddings
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Loading DINO model...")
        with open(config['dino_config'], 'r') as f:
            dino_model_args = yaml.safe_load(f)['student']
        dino_model, _ = get_dino_model(dino_model_args, config)
        dino_model = dino_model.to(device)
        process_embeddings(dino_model, None, dataloaders, device, 'dino', config, output_dir)
        
        # Clear DINO model from memory
        del dino_model
        torch.cuda.empty_cache()
    
    if model_type in ['both', 'clip']:
        # Get dataloaders for CLIP without preprocessing
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Loading DataLoaders for CLIP. no_transform=True...")
        dataloaders = dataloaders_from_config(config, no_transform=True)
        
        # Process CLIP embeddings
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Loading CLIP model...")
        clip_model, clip_preprocess = get_clip_model(None, config)
        clip_model = clip_model.to(device)
        process_embeddings(clip_model, clip_preprocess, dataloaders, device, 'clip', config, output_dir)
        
        # Clean up CLIP model
        del clip_model
        torch.cuda.empty_cache()
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - All embeddings generation completed")

if __name__ == "__main__":
    main() 