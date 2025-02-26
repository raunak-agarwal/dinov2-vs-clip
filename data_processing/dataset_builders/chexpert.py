"""
This script prepares the CheXpert dataset for training DINO, CLIP, and multi-label classification.
It resizes the images to 224x224 pixels and organizes the data for different tasks.

After processing, it creates two subdirectories within the output image directory:
    - DINO: Images only
        - dino/train, dino/val, dino/test 
    - CLIP: Images and reports
        - clip/train, clip/val, clip/test
It also creates a CSV file `chexpert_processed.csv` with the image paths, labels, and reports for each image.

Usage:
python chexpert.py --base_path /path/to/CheXpert --output_image_dir /path/to/output --existing_jpg_dir /path/to/existing_jpg_dir

python -i data_processing/dataset_builders/chexpert.py \
--base_path data/CheXpert --output_image_dir data/CheXpert/final-datasets/ \
--existing_jpg_dir data/MIMIC-CXR/mimic-cxr-jpg-2.0.0-small/final-datasets/dino/train ;
"""

import argparse
import os
import pandas as pd
import shutil
from tqdm import tqdm
from PIL import Image
import uuid
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--output_image_dir', type=str, required=True)
    parser.add_argument('--existing_jpg_dir', type=str, required=True)
    return parser.parse_args()

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def resize_image(input_path, output_path, size=(224, 224)):
    with Image.open(input_path) as img:
        img = img.resize(size, Image.BICUBIC)
        img.save(output_path)

def get_report_from_row(row):
    def replace_nan(t):
        return "" if pd.isna(t) else t
    
    section_impression = replace_nan(row['section_impression'])
    section_findings = replace_nan(row['section_findings'])
    section_summary = replace_nan(row['section_summary'])
    section_report = replace_nan(row['report'])    
    
    if section_impression and section_findings and section_summary:
        text = f"IMPRESSION: {section_impression}\nFINDINGS: {section_findings}\nSUMMARY: {section_summary}"
    elif section_impression and section_findings:
        text = f"IMPRESSION: {section_impression}\nFINDINGS: {section_findings}"
    elif section_impression:
        text = f"IMPRESSION: {section_impression}"
    elif section_findings:
        text = f"FINDINGS: {section_findings}"
    elif section_report:
        text = f"REPORT: {section_report}"
    else:
        text = ""
        
    text = " ".join(text.split()).strip().lower()
    
    return {'report': text}

def get_existing_uuids(directory):
    jpg_files = glob.glob(os.path.join(directory, '*.jpg'))
    existing_uuids = set()
    for jpg_file in jpg_files:
        try:
            file_uuid = jpg_file.split('/')[-1].split('.')[0]
            existing_uuids.add(str(file_uuid))
        except ValueError:
            continue # If the filename is not a valid UUID, skip it
    return existing_uuids

def generate_unique_uuid(existing_uuids):
    while True:
        new_uuid = str(uuid.uuid4())
        if new_uuid not in existing_uuids:
            existing_uuids.add(new_uuid)
            return new_uuid
        
        
def main(args):
    args = parse_args()
    print("Loading and processing data...")
    df = pd.read_csv(os.path.join(args.base_path, 'df_chexpert_plus_240401.csv'))
    print(f"Loaded {len(df)} records from df_chexpert_plus_240401.csv.")

    test_df = pd.read_csv(os.path.join(args.base_path, 'chexpert_5x200.csv'))
    print(f"Loaded {len(test_df)} records from chexpert_5x200.csv.")

    df['split'] = df['split'].apply(lambda x: 'validate' if x == 'valid' else x)
    df.drop(columns=['path_to_dcm', 'frontal_lateral', 'ap_pa'], inplace=True)

    test_df['Path'] = test_df['Path'].apply(lambda x: x.replace("Chexpert-v1.0", ""))
    df = df[~df['path_to_image'].isin(test_df['Path'])] # If the same path_to_image exists in both df and test_df, drop it from df

    # Get existing UUIDs
    existing_uuids = get_existing_uuids(args.existing_jpg_dir)
    print(f"Found {len(existing_uuids)} existing UUIDs in the directory.")

    # Generate unique UUIDs for each row
    df['uuid'] = df.apply(lambda _: generate_unique_uuid(existing_uuids), axis=1)

    # Create output directories
    for split in ['train', 'validate']:
        create_folder(os.path.join(args.output_image_dir, 'dino', split))
        create_folder(os.path.join(args.output_image_dir, 'clip', split))

    print("Removing empty reports...")
    df = df[df['report'].notna()]
    df = df[df['report'] != '']
    print("Empty reports removed.")

    print("Processing images for DINO...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        input_image_path = os.path.join(args.base_path, 'CheXpert-v1.0-small', row['path_to_image'])
        split = row['split']
        
        # Use UUID for the new filename
        new_filename = f"{row['uuid']}.jpg"
        
        # Process for DINO
        dino_output_path = os.path.join(args.output_image_dir, 'dino', split, new_filename)
        resize_image(input_image_path, dino_output_path)
        
    print("Copying and resizing images and reports for CLIP...")

    # Copy the dino folder structure to clip
    shutil.copytree(os.path.join(args.output_image_dir, 'dino'), os.path.join(args.output_image_dir, 'clip'), dirs_exist_ok=True)

    # Add report files for each image
    for split in ['train', 'validate']:
        for _, row in tqdm(df[df['split'] == split].iterrows(), total=df[df['split'] == split].shape[0], desc=f"Processing {split} split"):
            clip_image_path = os.path.join(args.output_image_dir, 'clip', split, f"{row['uuid']}.jpg")
            report = get_report_from_row(row)['report']
            report_path = os.path.splitext(clip_image_path)[0] + '.txt'
            with open(report_path, 'w') as f:
                f.write(report)

    print("Images and reports copied.")

    print("Saving processed data...")
    output_csv_path = os.path.join(args.base_path, 'chexpert_processed.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Processed data saved to {output_csv_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)