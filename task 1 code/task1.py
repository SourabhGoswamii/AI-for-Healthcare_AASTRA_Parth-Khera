import os
import pandas as pd
import nibabel as nib
import numpy as np
import subprocess
from nilearn import image, datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob


RAW_MRI_DIR = "MRI"
METADATA_CSV = "mri_metadata.csv"
OUTPUT_DIR = "processed_dataset"
LOG_FILE = "pipeline_log.csv"
TARGET_SHAPE = (128, 128, 128)

os.makedirs(OUTPUT_DIR, exist_ok=True)
for split in ['train', 'val', 'test']:
    for label in ['AD', 'MCI', 'CN']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)


def preprocess_nifti(input_path, output_path):
    """Executes the 4-step flowchart: Segment -> Register -> Crop -> Scale"""
    try:
        
        mni_template = datasets.load_mni152_template()
        resampled_img = image.resample_to_img(input_path, mni_template)
        
        
        gm_mask = datasets.load_mni152_gm_mask()
        gm_img = image.math_img("img * mask", img=resampled_img, mask=gm_mask)
        
        
        data = gm_img.get_fdata()
        c_x, c_y, c_z = np.array(data.shape) // 2
        t_x, t_y, t_z = np.array(TARGET_SHAPE) // 2
        cropped = data[c_x-t_x:c_x+t_x, c_y-t_y:c_y+t_y, c_z-t_z:c_z+t_z]
        
        
        denom = (cropped.max() - cropped.min()) + 1e-8
        processed = (cropped - cropped.min()) / denom
        
       
        final_img = nib.Nifti1Image(processed, mni_template.affine)
        nib.save(final_img, output_path)
        return True
    except Exception as e:
        return str(e)



print("Cleaning Metadata & Removing Duplicates...")
df = pd.read_csv(METADATA_CSV)
df.columns = [c.strip().lower() for c in df.columns]
df = df.drop_duplicates(subset=['subject'], keep='first')
label_map = dict(zip(df['subject'].astype(str), df['group']))

print(" Mapping Files and Checking Checkpoints...")

processed_files = glob.glob(f"{OUTPUT_DIR}/*/*/*.nii.gz")
processed_subjects = [os.path.basename(f).split('_')[0] for f in processed_files]

data_to_process = []
for subject_id in label_map.keys():
    if subject_id in processed_subjects:
        continue 
        

    subject_path = os.path.join(RAW_MRI_DIR, subject_id)
    if os.path.exists(subject_path):
        for root, _, files in os.walk(subject_path):
            if files:
                data_to_process.append({
                    'id': subject_id,
                    'dir': root,
                    'label': label_map[subject_id]
                })
                break

print(f" Found {len(data_to_process)} new subjects to process.")

if len(data_to_process) > 0:

    train, test = train_test_split(data_to_process, test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)
    
    splits = {'train': train, 'val': val, 'test': test}

    print(" Executing Preprocessing (Segmentation & Registration)...")
    for split_name, subjects in splits.items():
        for sub in tqdm(subjects, desc=f"Split: {split_name}"):
       
            temp_nii = f"temp_{sub['id']}.nii.gz"
            subprocess.run(f"dcm2niix -z y -o . -f temp_{sub['id']} {sub['dir']}", 
                           shell=True, capture_output=True)
            
            if os.path.exists(temp_nii):
                final_out = os.path.join(OUTPUT_DIR, split_name, sub['label'], f"{sub['id']}_proc.nii.gz")
                status = preprocess_nifti(temp_nii, final_out)
                
                os.remove(temp_nii)
                if status is not True:
                    print(f" Failed {sub['id']}: {status}")

print("Done")