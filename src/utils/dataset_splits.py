import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming you have a DataFrame 'df' with columns 'image', 'main_class',
# 'subclass'


path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular_datasetsplit.csv")
path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")
os.makedirs(path_splits, exist_ok=True)
df = pd.read_csv(path_classes)

ids = df['ID'].tolist()
index_of_106 = ids.index(106)
index_of_311 = ids.index(311)
df = df.drop(index_of_106)
df = df.drop(index_of_311)
## Split off the test set first
train, test = train_test_split(df, test_size=0.2, stratify=df['SUBCLASS'])

# Save the test set
test.to_csv(path_splits / "test_set_subclasses_2.csv", index=False)
train.to_csv(path_splits / "training_set_subclasses_2.csv", index=False)

# # Function to create and save three different stratified splits for training
# # and validation
# def create_and_save_splits(data):
#     for i in range(3):  # Create 3 different splits
#         # Shuffle the dataset to ensure randomness in splits
#         data = data.sample(frac=1).reset_index(drop=True)
#         # Stratified split
#         train, val = train_test_split(data, test_size=0.4, stratify=data['CLASS'])  # Adjust sizes as needed
#
#         # Save each split to CSV
#
#         val.to_csv(data_root / f'splits/validation_set_{i + 1}_classes_4.csv', index=False)
#
#
# # Create 3 stratified training/validation splits and save them
# create_and_save_splits(train_val)

# You have now saved: - 1 test set CSV: 'test_set.csv' - 3 training set CSVs:
# 'training_set_1.csv', 'training_set_2.csv', 'training_set_3.csv' - 3
# validation set CSVs: 'validation_set_1.csv', 'validation_set_2.csv',
# 'validation_set_3.csv'
