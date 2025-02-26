"""
Image Similarity Evaluation Module

This module provides functionality for computing and evaluating image similarities using deep learning embeddings.
Key components:

- Image embedding computation using ResNet50
- Similarity matrix calculation with efficient batching
- Evaluation metrics: Precision@k and Coverage@k
- Support for large-scale image datasets with memory-efficient processing

Main Functions:
- main(): Computes embeddings and similarity matrix for a set of images
- compute_similarity_matrix(): Efficiently computes pairwise similarities between embeddings
- evaluate_embeddings(): Calculates precision and coverage metrics at different k values
- compute_metrics_at_k(): Computes precision@k and coverage@k for individual queries

Metrics:
- Precision@k: Measures relevance of retrieved images based on label overlap
- Coverage@k: Measures how well retrieved images cover the query image's labels

Usage:
    similarity_matrix = main(image_paths)
    results = evaluate_embeddings(similarity_matrix, labels, k_values=[1, 5, 10])
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import PIL
from typing import List, Optional
from pathlib import Path
from dataloaders import get_dataset
import argparse
import yaml
import json
from typing import List, Optional

from modeling import DINOEmbeddingModel
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import _MODEL_CONFIGS

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='knn_config.yaml',
                      help='Path to knn config YAML file')
    parser.add_argument('--model_args', type=str, default='dinov2_vitb16.yaml',
                      help='Path to model args file')
    return parser.parse_args()

def get_model(model_args: dict, training_args: dict):
    if training_args['dino_or_clip'] == 'dino':
        model = DINOEmbeddingModel(model_args, training_args)
    elif training_args['dino_or_clip'] == 'clip':
        model = create_model_and_transforms(model_args['arch'], pretrained=model_args['model_path'])
    return model

def compute_similarity_matrix(embeddings: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
    num_embeddings = embeddings.shape[0]
    similarity_matrix = torch.zeros((num_embeddings, num_embeddings), 
                                 device=embeddings.device, 
                                 dtype=embeddings.dtype)
    
    # Pre-compute the norms if they haven't been normalized yet
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-4):
        embeddings = embeddings / norms

    for i in tqdm(range(0, num_embeddings, batch_size), desc="Computing similarities"):
        i_end = min(i + batch_size, num_embeddings)
        batch1 = embeddings[i:i_end]
        
        # Compute similarities for the entire row at once
        # This is more efficient than the nested loop in the original
        sim = torch.mm(batch1, embeddings.T)
        similarity_matrix[i:i_end] = sim

    return similarity_matrix

def compute_embeddings(dataloader, model, device):
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            images = batch[0]  # Assume first element contains images
            images = images.to(device)
            features = model(images)
            embeddings.append(features)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

def compute_relevance(query_labels: torch.Tensor, candidate_labels: torch.Tensor) -> float:
    """
    Compute relevance score between query and candidate labels using Jaccard similarity
    
    Args:
        query_labels: Binary tensor of shape (num_classes,) 
        candidate_labels: Binary tensor of shape (num_classes,)
    """
    intersection = torch.sum(query_labels & candidate_labels).float()
    union = torch.sum(query_labels | candidate_labels).float()
    
    # Handle edge case where union is 0
    if union == 0:
        return 0.0
    
    return intersection / union

def compute_metrics_at_k(
    similarity_matrix: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Precision@k and Coverage@k for each query in the dataset
    """
    num_queries = similarity_matrix.shape[0]
    precision_scores = torch.zeros(num_queries, device=device)
    coverage_scores = torch.zeros(num_queries, device=device)
    
    # Get top k+1 to account for the query image itself
    _, topk_indices = similarity_matrix.topk(k=k+1, dim=1)
    
    # Convert labels to bool type once
    labels = labels.bool()
    
    # Compute metrics for each query
    for query_idx in tqdm(range(num_queries), desc="Computing metrics@k"):
        query_labels = labels[query_idx]
        retrieved_indices = topk_indices[query_idx][1:k+1]  # Skip first result (query image)
        
        # Compute precision
        relevance_scores = torch.tensor([
            compute_relevance(query_labels, labels[candidate_idx])
            for candidate_idx in retrieved_indices
        ], device=device)
        precision_scores[query_idx] = torch.mean(relevance_scores)
        
        # Compute coverage
        num_query_labels = torch.sum(query_labels).float()
        if num_query_labels > 0:
            union_of_intersections = torch.zeros_like(query_labels, dtype=torch.bool)
            for candidate_idx in retrieved_indices:
                # Compute intersection for this candidate
                intersection = query_labels & labels[candidate_idx]  # Both are bool tensors 
                # Update union
                union_of_intersections |= intersection
            
            # Calculate coverage as |union of intersections| / |query labels|
            coverage_scores[query_idx] = (
                torch.sum(union_of_intersections).float() / 
                torch.sum(query_labels).float()
            )
        else:
            coverage_scores[query_idx] = 0.0
    
    return precision_scores, coverage_scores

def evaluate_embeddings(
    similarity_matrix: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Evaluate embedding quality using Precision@k and Coverage@k for multiple k values
    
    Args:
        similarity_matrix: Tensor of shape (num_queries, num_candidates)
        labels: Tensor of shape (num_queries, num_classes)
        k_values: List of k values to compute metrics for
    
    Returns:
        Dictionary containing mean Precision@k and Coverage@k scores for each k
    """
    results = {}
    
    # Move tensors to specified device
    similarity_matrix = similarity_matrix.to(device)
    labels = labels.to(device)
    
    for k in k_values:
        precision_scores, coverage_scores = compute_metrics_at_k(
            similarity_matrix, labels, k, device
        )
        results[f"precision@{k}"] = precision_scores.mean().item()
        results[f"coverage@{k}"] = coverage_scores.mean().item()
    
    return results

def main(
    knn_args: dict,
    batch_size: int = 32,
    embedding_batch_size: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    
    embeddings_path = knn_args.get('embeddings_path')
    model = get_model(model_cfg, knn_args)
    model.to(device)
    _, _, test_dataset = get_dataset(knn_args)
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    # Load pre-computed embeddings if they exist
    if embeddings_path is not None and Path(embeddings_path).exists():
        embeddings = torch.from_numpy(np.load(embeddings_path))
        print(f"Loaded pre-computed embeddings from {embeddings_path}")
        embeddings = embeddings.to(device)
    else:
        # Compute embeddings if not found
        embeddings = compute_embeddings(dataloader, model, device)
        # Save embeddings if path is provided
        if embeddings_path:
            np.save(embeddings_path, embeddings.cpu().numpy())
            print(f"Saved embeddings to {embeddings_path}")
    
    # Compute and return the similarity matrix
    return compute_similarity_matrix(embeddings, batch_size=embedding_batch_size)

if __name__ == "__main__":
    args = arg_parser()
    
    # Load training config
    with open(args.cfg, 'r') as file:
        knn_args = yaml.safe_load(file)
    
    # List all image file paths
    # image_dir = Path(knn_args['image_dir'])
    # image_paths = list(image_dir.glob("*.jpg"))
    
    # Compute the similarity matrix
    similarity_matrix = main(knn_args)
    
    # Load the labels from the pandas df
    labels = pd.read_csv(knn_args['label_file'], sep='\t')['label']
    labels = torch.from_numpy(labels.values)
    
    # Evaluate embeddings
    results = evaluate_embeddings(similarity_matrix, labels)
    
    # Print results
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")        
        
        
"""
import torch
import numpy as np
from tqdm import tqdm

# Create sample embeddings (5000 images, 512-dim embeddings)
embeddings = torch.randn(5000, 512)
embeddings = torch.nn.functional.normalize(embeddings, dim=1)  # normalize to unit vectors

# Create similarity matrix
similarity_matrix = torch.mm(embeddings, embeddings.t())

# Assign random labels using binomial distribution with p=0.2
# Make sure there is at least one 1 in each row
labels = torch.bernoulli(torch.full((5000, 10), 0.1)).long()  # Cast to long for bitwise ops

# Find rows with all zeros and add a 1 in a random column
zero_rows = (labels.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
if len(zero_rows) > 0:
    random_cols = torch.randint(0, 10, (len(zero_rows),))
    labels[zero_rows, random_cols] = 1

evaluate_embeddings(similarity_matrix, labels)

"""


""" 
import torch
from tqdm import tqdm

# Create larger test data for debugging
num_queries = 50
num_classes = 10
device = "cpu"

# Create a similarity matrix (20x20) using random embeddings
embeddings = torch.randn(num_queries, 128)  # Using 128-dim embeddings for this test
embeddings = torch.nn.functional.normalize(embeddings, dim=1)  # normalize to unit vectors
similarity_matrix = torch.mm(embeddings, embeddings.t())

# Generate random labels with p=0.2 probability of 1
labels = torch.bernoulli(torch.full((num_queries, num_classes), 0.2)).long()

# Find rows with all zeros and add a 1 in a random column
zero_rows = (labels.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
if len(zero_rows) > 0:
    random_cols = torch.randint(0, num_classes, (len(zero_rows),))
    labels[zero_rows, random_cols] = 1

k = 3  # Increased k for more neighbors
print(f"\nInitial setup:")
print(f"Similarity Matrix shape: {similarity_matrix.shape}")
print(f"Labels shape: {labels.shape}")
print(f"\nFirst few rows of labels:")
print(labels[:5])

# Initialize score tensors
precision_scores = torch.zeros(num_queries, device=device)
coverage_scores = torch.zeros(num_queries, device=device)

# Get top k+1 indices
_, topk_indices = similarity_matrix.topk(k=k+1, dim=1)
print(f"\nTop {k+1} indices for each query (first 5 rows):\n{topk_indices[:5]}")

# Convert labels to bool type
labels = labels.bool()

# Debug each query
for query_idx in range(num_queries):
    print(f"\n{'='*50}")
    print(f"Processing query {query_idx}")
    
    query_labels = labels[query_idx]
    print(f"Query labels: {query_labels}")
    print(f"Number of labels for query: {torch.sum(query_labels)}")
    
    # Get retrieved indices (skip first as it's the query itself)
    retrieved_indices = topk_indices[query_idx][1:k+1]
    print(f"Retrieved indices (excluding self): {retrieved_indices}")
    
    # Debug relevance computation
    relevance_scores = []
    for candidate_idx in retrieved_indices:
        candidate_labels = labels[candidate_idx]
        print(f"\nCandidate {candidate_idx} labels: {candidate_labels}")
        
        # Compute intersection and union
        intersection = torch.sum(query_labels & candidate_labels).float()
        union = torch.sum(query_labels | candidate_labels).float()
        relevance = intersection / union if union > 0 else 0.0
        
        print(f"Intersection: {intersection}")
        print(f"Union: {union}")
        print(f"Relevance score: {relevance:.4f}")
        
        relevance_scores.append(relevance)
    
    # Convert to tensor and compute precision
    relevance_scores = torch.tensor(relevance_scores, device=device)
    precision_scores[query_idx] = torch.mean(relevance_scores)
    print(f"\nPrecision score for query {query_idx}: {precision_scores[query_idx]:.4f}")
    
    # Debug coverage computation
    union_of_intersections = torch.zeros_like(query_labels, dtype=torch.bool)
    print(f"\nInitial union of intersections: {union_of_intersections}")
    
    for candidate_idx in retrieved_indices:
        intersection = query_labels & labels[candidate_idx]
        print(f"\nCandidate {candidate_idx}")
        print(f"Intersection with query: {intersection}")
        union_of_intersections |= intersection
        print(f"Updated union of intersections: {union_of_intersections}")
    
    coverage_score = (
        torch.sum(union_of_intersections).float() / 
        torch.sum(query_labels).float()
    )
    coverage_scores[query_idx] = coverage_score
    print(f"\nCoverage score for query {query_idx}: {coverage_scores[query_idx]:.4f}")

print(f"\n{'='*50}")
print("Final Results:")
print(f"Precision scores: {precision_scores}")
print(f"Mean precision@{k}: {precision_scores.mean():.4f}")
print(f"Coverage scores: {coverage_scores}")
print(f"Mean coverage@{k}: {coverage_scores.mean():.4f}")

# Additional statistics
print(f"\nStatistics:")
print(f"Average number of labels per image: {torch.sum(labels).item() / num_queries:.2f}")
print(f"Label distribution across classes:")
for class_idx in range(num_classes):
    count = torch.sum(labels[:, class_idx]).item()
    print(f"Class {class_idx}: {count} images ({count/num_queries*100:.1f}%)") 

"""
