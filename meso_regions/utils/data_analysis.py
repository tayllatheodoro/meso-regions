import nibabel as nib
import pandas as pd
import os
from pathlib import Path

# Directory where your MRI images are stored
image_directory = Path('/data_lids/home/taylla/PycharmProjects/meso/data/resample/images_224_224/images')
output_directory = Path('/data_lids/home/taylla/PycharmProjects/meso/output/data_overview/resample_dims')
os.makedirs(output_directory, exist_ok=True)
# List to hold image data
image_data = []

# Loop through all the files in the directory
for filename in os.listdir(image_directory):
    if  filename.endswith(
            '.nii.gz'):  # Adjust based on your file type
        file_path =image_directory/filename
        img = nib.load(file_path)
        header = img.header

        # Extracting additional details
        dimensions = header.get_data_shape()  # Spatial and temporal dimensions
        voxel_spacing = header.get_zooms()  # Voxel dimensions in mm and time spacing
        data_type = str(header.get_data_dtype())
        file_size = os.path.getsize(file_path)  # File size in bytes

        # Append the data to the list
        image_data.append({
            'Filename': filename,
            'Dimensions': dimensions,
            'Voxel Spacing': voxel_spacing,
            'Data Type': data_type,
            'File Size (bytes)': file_size,

        })

# Create a DataFrame
df = pd.DataFrame(image_data)

# Save DataFrame to a CSV file
df.to_csv(output_directory/'MRI_image_characteristics.csv', index=False)

print("Data extraction complete. CSV file has been saved.")

# Create summary tables for unique counts of dimensions and voxel spacing
dimension_counts = df['Dimensions'].astype(str).value_counts().reset_index()
dimension_counts.columns = ['Dimensions', 'Count']

voxel_spacing_counts = df['Voxel Spacing'].astype(str).value_counts().reset_index()
voxel_spacing_counts.columns = ['Voxel Spacing', 'Count']

# Save summary tables to CSV files
dimension_counts.to_csv(output_directory / 'dimension_counts.csv', index=False)
voxel_spacing_counts.to_csv(output_directory / 'voxel_spacing_counts.csv', index=False)

print("Data extraction and summary tables have been saved.")
