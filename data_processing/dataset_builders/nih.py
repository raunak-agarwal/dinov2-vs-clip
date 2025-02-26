import json
import tarfile
from PIL import Image
from tqdm import tqdm
import argparse
import os
import pandas as pd
import random
import shutil
import datetime
import glob
import uuid

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--label-to-text-file', type=str, required=True)
    return parser.parse_args()

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def resize_image(input_path, output_path, size=(224, 224)):
    with Image.open(input_path) as img:
        img = img.resize(size, Image.BICUBIC)
        img = img.convert('RGB')  # Ensure the image is in RGB mode
        img.save(output_path, 'JPEG')

def create_report(row):
    global label_to_text_dict, labels
    positive_labels = []
    negative_labels = []
    for i, label in enumerate(labels):
        if row.iloc[i+1] == 1:  # Use iloc for positional indexing
            positive_labels.append(label)
        else:
            negative_labels.append(label)
            
    if "No Finding" in negative_labels:
        negative_labels.remove("No Finding")
    if "No Finding" in positive_labels:
        positive_labels.remove("No Finding")
        
    pos_texts = []
    neg_texts = []
    for label in positive_labels:
        pos_texts.append(random.choice(label_to_text_dict[label]['pos']))
    for label in negative_labels:
        neg_texts.append(random.choice(label_to_text_dict[label]['neg']))
        
    neg_texts = random.sample(neg_texts, random.randint(4, 8))
    text = pos_texts + neg_texts
    text = " ".join(text).strip().lower()
    row = row.copy()  # Ensure we are working with a copy
    row['report'] = text
    return row  # Return the modified row

def process_split(df, split, base_path, output_path):
    print(f"Processing {split} split...")
    
    dino_output_dir = os.path.join(output_path, 'dino', split)
    clip_output_dir = os.path.join(output_path, 'clip', split)
    create_folder(dino_output_dir)
    create_folder(clip_output_dir)
    
    new_ids = []
    new_paths = []
    
    # Step 1: Process images for DINO and CLIP
    print("Processing images for DINO and CLIP...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_filename = row['id']
        input_image_path = os.path.join(base_path, 'images', 'files', image_filename)
        
        # Generate new ID based on UUID
        new_id = f"nih_{uuid.uuid4()}"
        new_ids.append(new_id)
        
        # Resize, convert to jpg, and rename for DINO
        dino_jpg_path = os.path.join(dino_output_dir, f"{new_id}.jpg")
        resize_image(input_image_path, dino_jpg_path)
        
        # Copy to CLIP
        clip_jpg_path = os.path.join(clip_output_dir, f"{new_id}.jpg")
        shutil.copy(dino_jpg_path, clip_jpg_path)
        
        # Store new path
        new_paths.append(os.path.join('clip', split, f"{new_id}.jpg"))
    
    # Step 2: Create report files for CLIP and apply create_report
    print("Creating report files for CLIP and applying create_report...")
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        new_id = new_ids[i]
        clip_jpg_path = os.path.join(clip_output_dir, f"{new_id}.jpg")
        
        # Create report file for CLIP
        report_path = os.path.splitext(clip_jpg_path)[0] + '.txt'
        
        # Apply create_report function
        row = create_report(row)
        
        with open(report_path, 'w') as f:
            f.write(row['report'])
        
        # Update the dataframe with the new report
        df.loc[i] = row  # Use loc to update the DataFrame

    # Update dataframe with new IDs and paths
    df['id'] = new_ids
    df['path'] = new_paths

    # Save processed dataframe
    df.to_csv(os.path.join(output_path, f'{split}.csv'), index=False)

    return df

# def main():
args = parse_args()

global label_to_text_dict, labels
label_to_text_dict = json.load(open(args.label_to_text_file, 'r'))

# In the dictionary, replace the key "Pleural Effusion" with "Effusion"
label_to_text_dict["Effusion"] = label_to_text_dict.pop("Pleural Effusion")

# Replace Pleural_Thickening with Pleural Thickening  
label_to_text_dict["Pleural Thickening"] = label_to_text_dict.pop("Pleural_Thickening")


labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
            "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", 
            "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax", 
            "Pneumoperitoneum", "Pneumomediastinum", "Subcutaneous Emphysema", 
            "Tortuous Aorta", "Calcification of the Aorta", "No Finding"]

cols = ['id'] + labels

prunecxr_train = pd.read_csv(os.path.join(args.base_path, 'PruneCXR/miccai2023_nih-cxr-lt_labels_train.csv'))[cols]
prunecxr_val = pd.read_csv(os.path.join(args.base_path, 'PruneCXR/miccai2023_nih-cxr-lt_labels_val.csv'))[cols]
prunecxr_test = pd.read_csv(os.path.join(args.base_path, 'PruneCXR/miccai2023_nih-cxr-lt_labels_test.csv'))[cols]

# Replace the apply calls with normal loops using tqdm
print("Applying create_report to train split...")
for i in tqdm(range(len(prunecxr_train))):
    prunecxr_train.loc[i] = create_report(prunecxr_train.iloc[i].copy())  # Use loc to update the DataFrame

print("Applying create_report to validation split...")
for i in tqdm(range(len(prunecxr_val))):
    prunecxr_val.loc[i] = create_report(prunecxr_val.iloc[i].copy())  # Use loc to update the DataFrame

print("Applying create_report to test split...")
for i in tqdm(range(len(prunecxr_test))):
    prunecxr_test.loc[i] = create_report(prunecxr_test.iloc[i].copy())  # Use loc to update the DataFrame

# Process each split
prunecxr_train = process_split(prunecxr_train, 'train', args.base_path, args.output_path)
prunecxr_val = process_split(prunecxr_val, 'val', args.base_path, args.output_path)
prunecxr_test = process_split(prunecxr_test, 'test', args.base_path, args.output_path)

# Combine all processed dataframes
combined_df = pd.concat([prunecxr_train, prunecxr_val, prunecxr_test], axis=0)

# Save the combined processed dataframe
combined_df.to_csv(os.path.join(args.output_path, 'nih_processed.csv'), index=False)

