import os
import pandas as pd


METADATA_PATH = 'mri_metadata_final_cleaned.csv'
df = pd.read_csv(METADATA_PATH)

id_to_label = dict(zip(df['subject'].astype(str).str.strip(), df['group']))


mapped_data = []
BASE_DIR = 'MRI'

print("🔍 Mapping files to labels...")

for subject_id in id_to_label.keys():
    subject_path = os.path.join(BASE_DIR, subject_id)
    
    if os.path.exists(subject_path):
      
        for root, dirs, files in os.walk(subject_path):
            for file in files:
                
                full_path = os.path.join(root, file)
                
                
                if not file.startswith('.'):
                    mapped_data.append({
                        'file_path': full_path,
                        'subject': subject_id,
                        'label': id_to_label[subject_id],
                        'extension': os.path.splitext(file)[1]
                    })

mapping_df = pd.DataFrame(mapped_data)
mapping_df.to_csv('final_file_mapping.csv', index=False)

print(f"✅ Mapping Complete!")
print(f"Total files mapped: {len(mapping_df)}")
print(f"Sample mapping:\n{mapping_df[['subject', 'label', 'extension']].head(3)}")