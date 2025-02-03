import os
from pathlib import Path
import pandas as pd

def prepare_data(data_dir):
    path_d = Path(data_dir)
    img_path_list = sorted([str(item) for item in path_d.rglob('*.png')])
    label_path_list = sorted([str(item) for item in path_d.rglob('*.txt')])

    def get_text(one_label_path):
        with open(one_label_path, 'r', encoding='gbk') as f:
            return f.readlines()[0].strip()

    label_name_list = [get_text(path) for path in label_path_list]
    label_mapping = {'Normal':0, 'Mild':1, 'Moderate':2, 'Severe':3}
    label_num_list = [label_mapping.get(label, -1) for label in label_name_list]

    return pd.DataFrame({
        'img_path': img_path_list,
        'label_name': label_name_list,
        'label': label_num_list,
        'text': ['Example image'] * len(label_num_list)
    })

def save_prepared_data(train_dir, val_dir, output_dir):
    train_pd = prepare_data(train_dir)
    val_pd = prepare_data(val_dir)
    os.makedirs(output_dir, exist_ok=True)
    train_pd.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    val_pd.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)

if __name__ == "__main__":
    save_prepared_data(
        'data/raw',
        'data/processed',
        'data/processed_data'
    )
