import zipfile
import os
import pydicom
import cv2
import gc
from tqdm import tqdm
from PIL import Image
import argparse
from multiprocessing import Pool, cpu_count

def resize_image(input_path, output_path, size=(224, 224)):
    """
    Resize an image using PIL's BICUBIC interpolation.
    
    Args:
        input_path: Path to input image or a PIL Image object
        output_path: Path to save the resized image
        size: Tuple of (width, height) for the output image
    """
    with Image.open(input_path) as img:
        img = img.resize(size, Image.BICUBIC)
        img = img.convert('RGB')  # Ensure the image is in RGB mode
        img.save(output_path, 'JPEG')

def convert_dicom_to_cv2(dcm_file, temp_jpg_path):
    """
    Convert a DICOM file to a CV2 image and save it as a JPEG.
    
    Args:
        dcm_file: A file-like object containing DICOM data
        temp_jpg_path: Path to save the temporary JPEG image
    """
    dcm = pydicom.dcmread(dcm_file)
    rescaled_image = cv2.convertScaleAbs(dcm.pixel_array, alpha=(255.0 / dcm.pixel_array.max()))

    # Handle MONOCHROME1
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        rescaled_image = cv2.bitwise_not(rescaled_image)

    # Apply histogram equalization
    adjusted_image = cv2.equalizeHist(rescaled_image)

    # Save the adjusted image as a temporary JPEG
    cv2.imwrite(temp_jpg_path, adjusted_image)

def process_single_file(args):
    """
    Process a single DICOM file from the zip archive.
    
    Args:
        args: Tuple of (filename, zip_file_path, output_dir, size)
    """
    filename, zip_file_path, output_dir, size = args
    try:
        jpg_path = os.path.join(output_dir, filename.replace('.dicom', '.jpg'))
        temp_jpg_path = os.path.join(output_dir, 'temp', filename.replace('.dicom', '.jpg'))
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            with zip_ref.open(filename) as dcm_file:
                # Convert DICOM to CV2 Image and save as temporary JPEG
                convert_dicom_to_cv2(dcm_file, temp_jpg_path)
                
                # Resize using PIL and replace the original JPEG
                resize_image(temp_jpg_path, jpg_path, size=size)
                
                # Remove the temporary JPEG file
                os.remove(temp_jpg_path)
        
        return True
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return False

def process_zip_file(zip_file_path, output_dir, size=(224, 224), debug=False, num_workers=None):
    """
    Process a zip file containing DICOM files and convert them to resized JPEG images.
    
    Args:
        zip_file_path: Path to the input zip file
        output_dir: Directory to save processed images
        size: Output image size as (width, height)
        debug: If True, process only first 10 images
        num_workers: Number of worker processes. If None, uses CPU count - 1
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)    
    os.makedirs(os.path.join(output_dir, 'temp'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'temp', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'temp', 'test'), exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()    
        dicom_files = [f for f in all_files if f.endswith('.dicom')]
        if debug:
            dicom_files = dicom_files[:100]

        process_args = [(f, zip_file_path, output_dir, size) for f in dicom_files]
        
        # Use provided num_workers or default to CPU count - 1
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        with Pool(processes=num_workers) as pool:
            list(tqdm(
                pool.imap(process_single_file, process_args),
                total=len(dicom_files),
                desc="Converting DICOM files",
                unit="file"
            ))

def main():
    parser = argparse.ArgumentParser(description='Convert DICOM files from a zip archive to JPEG images')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to the input zip file containing DICOM images')
    parser.add_argument('--output', type=str, required=True,
                      help='Directory to save the output JPEG images')
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224],
                      help='Output image size as width height (default: 224 224)')
    parser.add_argument('--debug', action='store_true',
                      help='Process only first 100 images for debugging')
    parser.add_argument('--num-workers', type=int, default=None,
                      help='Number of worker processes. Defaults to CPU count - 1')
    
    args = parser.parse_args()
    
    process_zip_file(
        zip_file_path=args.input,
        output_dir=args.output,
        size=tuple(args.size),
        debug=args.debug,
        num_workers=args.num_workers
    )

if __name__ == '__main__':
    main()