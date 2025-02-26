"""
This script prepares the MIMIC-CXR dataset for training DINO, CLIP, and multi-label classification.
It resizes the images to 224x224 pixels. 

After resizing, it pulls the reports from the MIMIC-CXR-2.0.0-sectioned.csv file (this was done separately).
It then creates two subdirectories within the output image directory. 
    - DINO: Images only
        - dino/train, dino/val, dino/test 
    - CLIP: Images and reports
        - clip/train, clip/val, clip/test
It also creates a CSV `mimic_cxr_labels_reports_splits.csv` file with the image paths, labels, and reports for each image.

Usage:
python mimic.py --base_path /path/to/mimic-cxr --output_image_dir /path/to/output
"""

import glob
import argparse
import os
from tqdm import tqdm
import pandas as pd
import shutil
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--output_image_dir', type=str, required=True)
    return parser.parse_args()

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_path(image_dir, dicom_id, study_id, subject_id):
    img_path = os.path.join(image_dir, f"p{subject_id[:2]}", 
                        f"p{subject_id}", f"s{study_id}", 
                        f"{dicom_id}.jpg")
    return img_path

def resize_image(input_path, size):
    with Image.open(input_path) as img:
        img = img.resize(size, Image.BICUBIC)
        img.save(input_path)  # Overwrite the original image

def get_report_from_df(row):
    def replace_nan(t):
        return "" if pd.isna(t) else t
    
    row['impression'] = replace_nan(row['impression'])
    row['findings'] = replace_nan(row['findings'])
    row['last_paragraph'] = replace_nan(row['last_paragraph'])
    row['comparison'] = replace_nan(row['comparison'])
    
    if row['impression'] and row['findings']:
        text = f"impression: {row['impression']}\nfindings: {row['findings']}"
    elif row['impression']:
        text = f"impression: {row['impression']}"
    elif row['findings']:
        text = f"findings: {row['findings']}"
    elif row['last_paragraph']:
        text = f"{row['last_paragraph']}"
    elif row['comparison']:
        text = f"{row['comparison']}"
    else:
        text = ""

    text = " ".join(text.split()).strip().lower()
    
    row['report'] = text
    return row

def copy_image(img_path, output_dir):
    shutil.copy(img_path, output_dir)


def main(args):
    args = parse_args()
    base_path = args.base_path
    print("Base path set to:", base_path)

    split_file = base_path + '/mimic-cxr-2.0.0-split.csv'
    label_file = base_path + '/mimic-cxr-2.0.0-chexpert.csv'
    image_dir = base_path + '/images/files'

    print("Loading split data...")
    splits_df = pd.read_csv(split_file)
    print("Loading label data...")
    label_df = pd.read_csv(label_file)
    label_df.fillna(0, inplace=True) # fill missing values with 0
    label_df.replace(-1, 1, inplace=True) # replace -1 with 1
    print("Label data processed.")

    print("Merging label and split data...")
    data = pd.merge(label_df, splits_df, on=['subject_id', 'study_id'], how='inner') 
    print("Data merged. Number of records:", data.shape[0])

    data[['subject_id']] = data[['subject_id']].astype(str) # Convert 'subject_id' to string
    data[['study_id']] = data[['study_id']].astype(str) # Convert 'study_id' to string

    print("Generating image paths...")
    data['img_path'] = data.apply(lambda x: get_path(image_dir, x['dicom_id'], x['study_id'], x['subject_id']), axis=1)
    print("Image paths generated.")

    print("Loading reports...")
    reports = pd.read_csv(base_path+"/mimic_cxr_sectioned.csv") # ['study', 'impression', 'findings', 'last_paragraph', 'comparison']
    reports['study'] = reports['study'].apply(lambda x: x.replace('s', ''))
    print("Reports loaded.")

    print("Merging data with reports...")
    data = pd.merge(data, reports, left_on='study_id', right_on='study', how='inner')
    print("Data merged with reports. Number of records:", data.shape[0])

    print("Generating reports...")
    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        data.at[i, 'report'] = get_report_from_df(row)['report']
    print("Reports generated.")

    print("Dropping unnecessary columns...")
    data.drop(columns=['study', 'impression', 'findings', 'last_paragraph', 'comparison'], inplace=True)
    print("Columns dropped.")

    print("Removing empty reports...")
    data = data[data['report'].notna()]
    data = data[data['report'] != '']
    print("Empty reports removed.")

    print("Saving final data to CSV...")
    data.to_csv(base_path + '/mimic_cxr_labels_reports_splits.csv', index=False)
    print("Data saved successfully.")
    print(f"Shape of the final dataset: {data.shape}")
    print("Process completed.")
    
    print("Copying and resizing images for dinov2...")
    create_folder(args.output_image_dir + '/dino/train')
    create_folder(args.output_image_dir + '/dino/validate')
    create_folder(args.output_image_dir + '/dino/test')
    for i, row in tqdm(data[data['split'] == 'train'].iterrows(), total=data[data['split'] == 'train'].shape[0]):
        copy_image(row['img_path'], args.output_image_dir + '/dino/train')
        resize_image(args.output_image_dir + '/dino/train/' + row['dicom_id'] + '.jpg', (224, 224))
    for i, row in tqdm(data[data['split'] == 'validate'].iterrows(), total=data[data['split'] == 'validate'].shape[0]):
        copy_image(row['img_path'], args.output_image_dir + '/dino/validate')
        resize_image(args.output_image_dir + '/dino/validate/' + row['dicom_id'] + '.jpg', (224, 224))
    for i, row in tqdm(data[data['split'] == 'test'].iterrows(), total=data[data['split'] == 'test'].shape[0]):
        copy_image(row['img_path'], args.output_image_dir + '/dino/test')
        resize_image(args.output_image_dir + '/dino/test/' + row['dicom_id'] + '.jpg', (224, 224))
    print("Images copied.")
    
    print("Copying and resizing images and reports for clip...")
    create_folder(args.output_image_dir + '/clip/train')
    create_folder(args.output_image_dir + '/clip/validate')
    create_folder(args.output_image_dir + '/clip/test')
    
    # Copy the dino folder structure to clip
    shutil.copytree(args.output_image_dir + '/dino', args.output_image_dir + '/clip', dirs_exist_ok=True)
    
    # Add report files for each image
    for split in ['train', 'validate', 'test']:
        for i, row in tqdm(data[data['split'] == split].iterrows(), total=data[data['split'] == split].shape[0], desc=f"Processing {split} split"):
            with open(args.output_image_dir + f'/clip/{split}/' + row['dicom_id'] + '.txt', 'w') as f:
                f.write(row['report'])
    
    print("Images and reports copied.")

if __name__ == "__main__":
    args = parse_args()
    main(args)