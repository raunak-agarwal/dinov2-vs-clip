import pandas as pd
import os
from PIL import Image
import argparse
import glob
from tqdm import tqdm
import shutil

def merge_indiana_data(base_path):
    projections_path = os.path.join(base_path, 'indiana_projections.csv')
    reports_path = os.path.join(base_path, 'indiana_reports.csv')
    
    projections = pd.read_csv(projections_path)
    reports = pd.read_csv(reports_path)
    
    # Merge reports with projections
    merged = pd.merge(projections, reports, on='uid', how='left')
    
    # Keep only necessary columns
    merged = merged[['uid', 'filename', 'projection', 'findings', 'impression']]
    merged['filename'] = merged['filename'].str.replace('.dcm.png', '.png')
    
    # Add "CXR" to the beginning of the filenames
    merged['filename'] = 'CXR' + merged['filename']
    
    return merged

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def resize_image(input_path, output_path, size=(224, 224)):
    with Image.open(input_path) as img:
        img = img.resize(size, Image.BICUBIC)
        img.save(output_path.replace('.png', '.jpg'), 'JPEG')

def process_indiana_images(base_path, output_path, image_size):
    merged_data = merge_indiana_data(base_path)
    input_image_dir = os.path.join(base_path, 'NLMCXR_png')
    
    # Create output directories
    create_folder(os.path.join(output_path, 'dino', 'train'))
    create_folder(os.path.join(output_path, 'clip', 'train'))
    
    processed_data = []
    
    for _, row in tqdm(merged_data.iterrows(), total=len(merged_data), desc="Processing images"):
        input_image_path = os.path.join(input_image_dir, row['filename'])
        
        if os.path.exists(input_image_path):
            # Generate report
            report = f"impression: {row['impression']}\nfindings: {row['findings']}"
            if report == "":
                continue
            
            # Process for DINO
            dino_output_path = os.path.join(output_path, 'dino', 'train', row['filename'].replace('.png', '.jpg'))
            resize_image(input_image_path, dino_output_path, image_size)
            
            # Process for CLIP
            clip_output_path = os.path.join(output_path, 'clip', 'train', row['filename'].replace('.png', '.jpg'))
            shutil.copy(dino_output_path, clip_output_path)
            
            # Delete the original PNG image
            os.remove(input_image_path)
            
            report_path = os.path.splitext(clip_output_path)[0] + '.txt'
            with open(report_path, 'w') as f:
                f.write(" ".join(report.split()).strip().lower())
            
            processed_data.append({
                'filename': row['filename'].replace('.png', '.jpg'),
                'projection': row['projection'],
                'report': report
            })
        else:
            print(f"Image not found: {input_image_path}")
    
    return processed_data

def main():
    parser = argparse.ArgumentParser(description="Process Indiana chest X-ray dataset for DINO and CLIP")
    parser.add_argument("--base_path", type=str, required=True, help="Base directory containing the dataset files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed datasets")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], help="Image size to resize to (width height)")
    
    args = parser.parse_args()
    
    processed_data = process_indiana_images(args.base_path, args.output_path, tuple(args.image_size))
    
    print(f"Processed {len(processed_data)} images")
    
    # Save processed data to CSV
    output_csv_path = os.path.join(args.output_path, 'nlmcxr_processed.csv')
    pd.DataFrame(processed_data).to_csv(output_csv_path, index=False)
    print(f"Processed data saved to {output_csv_path}")

if __name__ == "__main__":
    main()

