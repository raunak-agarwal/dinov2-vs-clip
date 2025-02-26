#!/usr/bin/env python3
"""
Script to package image-text pairs into tar files and a zip archive.

Takes a folder containing matching .jpg and .txt files (e.g., 1.jpg and 1.txt),
either uses all pairs or randomly samples N pairs, splits them into M tar files,
and creates a zip containing all tars. The script ensures thorough randomization
at multiple levels while maintaining image-text pair relationships.

Usage:
    python package_files.py --input-dir /path/to/input --output-dir /path/to/output 
                           [--num-tars 2176] [--sample-size 30000 | --use-all]

Arguments:
    --input-dir: Directory containing the .jpg and .txt files
    --output-dir: Directory where tar files and zip archive will be saved
    --num-tars: Number of tar files to create (default: 2176)
    --sample-size: Number of file pairs to randomly sample
    --use-all: Use all file pairs instead of sampling

Example:
1. With sampling
python clip-tar-prepare.py --input-dir /path/to/input --output-dir /path/to/output --num-tars 2176 --sample-size 30000

2. Using all files
python clip-tar-prepare.py --input-dir /path/to/input --output-dir /path/to/output --num-tars 2176 --use-all
"""

import os
import tarfile
import zipfile
import random
import argparse
from math import ceil
from tqdm import tqdm
import uuid
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Package image-text pairs into tar and zip archives')
    parser.add_argument('--input-dir', required=True, help='Input directory containing jpg/txt pairs')
    parser.add_argument('--output-dir', required=True, help='Output directory for tar/zip files')
    parser.add_argument('--num-tars', type=int, default=2176, help='Number of tar files to create')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sample-size', type=int, help='Number of pairs to sample')
    group.add_argument('--use-all', action='store_true', help='Use all file pairs instead of sampling')
    
    return parser.parse_args()

def get_file_pairs(input_dir):
    """Find and group jpg/txt files into pairs."""
    print("Finding and grouping file pairs...")
    all_files = [f for f in os.listdir(input_dir) if f.endswith((".jpg", ".txt"))]
    random.shuffle(all_files)
    
    # print the shuffled list and show the first 100 items
    print("Shuffled list:")
    print(all_files[:100])
    print()
        
    file_pairs = {}
    for file in tqdm(all_files, desc="Processing files"):
        name, ext = os.path.splitext(file)
        if name not in file_pairs:
            file_pairs[name] = []
        file_pairs[name].append(file)
    
    valid_pairs = {k: v for k, v in file_pairs.items() if len(v) == 2}
    print(f"Found {len(valid_pairs)} valid image-text pairs")
    return valid_pairs

def copy_and_rename_pairs(input_dir, work_dir, valid_pairs):
    """Copy and rename pairs with UUIDs one at a time."""
    new_files = []  # Will store tuples of (jpg_file, txt_file)
    used_uuids = set()
    
    print("Copying and renaming files with UUIDs...")
    for base_name, pair in tqdm(valid_pairs.items(), desc="Processing pairs"):
        while True:
            new_uuid = str(uuid.uuid4())
            if new_uuid not in used_uuids:
                used_uuids.add(new_uuid)
                break
        
        # Sort pair to ensure jpg comes first, txt second
        pair.sort(key=lambda x: os.path.splitext(x)[1])
        jpg_file, txt_file = None, None
        
        for file in pair:
            ext = os.path.splitext(file)[1]
            new_name = f"{new_uuid}{ext}"
            src = os.path.join(input_dir, file)
            dst = os.path.join(work_dir, new_name)
            shutil.copy2(src, dst)
            
            if ext == '.jpg':
                jpg_file = new_name
            elif ext == '.txt':
                txt_file = new_name
        
        new_files.append((jpg_file, txt_file))
    
    return new_files, len(valid_pairs)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Create temporary working directory
    work_dir = "clip_prepare"
    os.makedirs(work_dir, exist_ok=True)

    try:
        # Get valid pairs from input directory
        valid_pairs = get_file_pairs(args.input_dir)

        # Copy and rename files with UUIDs
        file_pairs, total_pairs = copy_and_rename_pairs(args.input_dir, work_dir, valid_pairs)
        
        # Sample pairs if needed
        if args.use_all:
            selected_pairs = file_pairs
            print(f"Using all {total_pairs} file pairs")
        else:
            if total_pairs < args.sample_size: 
                raise ValueError(f"Not enough file pairs to sample. Found {total_pairs} pairs, "
                               f"requested {args.sample_size}")
            selected_pairs = random.sample(file_pairs, args.sample_size)
            print(f"Randomly sampled {args.sample_size} pairs from {total_pairs} total pairs")

        random.shuffle(selected_pairs)
        
        # Distribute pairs across tars (keeping pairs together)
        chunk_size = ceil(len(selected_pairs) / args.num_tars)
        pair_chunks = [selected_pairs[i:i + chunk_size] 
                      for i in range(0, len(selected_pairs), chunk_size)]

        print("Creating tar files...")
        tar_files = []
        for i, pair_chunk in tqdm(enumerate(pair_chunks), total=len(pair_chunks), desc="Creating tars"):
            tar_name = f"{i:04d}.tar"
            tar_path = os.path.join(args.output_dir, tar_name)
            tar_files.append(tar_path)
            with tarfile.open(tar_path, "w") as tar:
                random.shuffle(pair_chunk)  # Shuffle pairs within the chunk
                for jpg_file, txt_file in pair_chunk:
                    for file_name in (jpg_file, txt_file):
                        file_path = os.path.join(work_dir, file_name)
                        tar.add(file_path, arcname=file_name)

        print("Creating final zip archive...")
        zip_path = os.path.join(args.output_dir, "train-data.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            random.shuffle(tar_files)
            for tar_file in tqdm(tar_files, desc="Adding to zip"):
                zipf.write(tar_file, arcname=os.path.basename(tar_file))

    finally:
        # Cleanup
        print("Cleaning up temporary files...")
        shutil.rmtree(work_dir)

    print(f"Successfully created {len(tar_files)} tar files and zip archive in: {args.output_dir}")

if __name__ == "__main__":
    main()