import wget, gdown, os
import zipfile
import json
from huggingface_hub import snapshot_download
import pandas as pd
from appdata import root, path

def download_files():
    if not os.path.exists(path):
        os.mkdir(path)

    # Download the COCO dataset
    file_links = json.load(open(f'{root}/config/file_links.json'))

    # Annotations file
    if not os.path.exists(f'{path}/captions_train2014.json'):
        print("Downloading annotation file...")
        wget.download(file_links['captions_train2014'], f'{path}/captions_train2014.json', bar=wget.bar_adaptive)
    
    # Instances file
    if not os.path.exists(f'{path}/captions_val2014.json'):
        print("Downloading annotation file...")
        gdown.download(file_links['captions_val2014'], f'{path}/captions_val2014.json', quiet=False)

    # Karpathy splits
    if not os.path.exists(f'{path}/karpathy_coco_for_eval.json'):
        print("Downloading splits file...")
        gdown.download(file_links['karpathy_coco'], f'{path}/karpathy_coco_for_eval.json', quiet=False)

    # Download the COCO2014 dataset
    if not os.path.exists(f'{path}/train2014'):
        print("Downloading MSCOCO 2014 dataset...")
        wget.download(file_links['MSCOCO2014'], f'{path}/train2014.zip', bar=wget.bar_adaptive)
        with zipfile.ZipFile(f'{path}/train2014.zip',"r") as zip_ref:
            zip_ref.extractall(f"{path}")
        os.remove(f'{path}/train2014.zip')

    if not os.path.exists(f'{path}/val2014'):
        print("Downloading MSCOCO 2014 dataset...")
        wget.download(file_links['MSCOCO2014_val'], f'{path}/val2014.zip', bar=wget.bar_adaptive)
        with zipfile.ZipFile(f'{path}/val2014.zip',"r") as zip_ref:
            zip_ref.extractall(f"{path}")
        os.remove(f'{path}/val2014.zip')

    # Download the rices features for scorer
    if not os.path.exists(f'{path}/RICES-features'):
        os.mkdir(f'{path}/RICES-features')
        print("Downloading rices features...")
        gdown.download(file_links['rices_features'], f'{path}/RICES-features/coco.pkl', quiet=False)

    # Download the model 
    #if not os.path.exists(f'{root}/checkpoints/OTTER-Image-MPT7B'):
    print("Downloading OTTER-Image-MPT7B...")
    snapshot_download(local_dir=f'{root}/checkpoints/OTTER-Image-MPT7B', repo_id='luodian/OTTER-Image-MPT7B', resume_download=True)

def make_dataset():
    print("Downloading files...")
    download_files()
    print("Done!")

if __name__ == '__main__':
    make_dataset()


