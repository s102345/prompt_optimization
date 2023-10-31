import wget, gdown, os
import zipfile
import json
from huggingface_hub import hf_hub_download
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
    if not os.path.exists(f'{path}/instances_train2014.json'):
        print("Downloading instance file...")
        gdown.download(file_links['instances_train2014'], f'{path}/instances_train2014.json', quiet=False)

    # Karpathy splits with modification
    if not os.path.exists(f'{path}/prompt_karpathy_coco.json'):
        print("Downloading splits file...")
        gdown.download(file_links['prompt_karpathy_coco'], f'{path}/prompt_karpathy_coco.json', quiet=False)

    # Download the COCO2014 dataset
    if not os.path.exists(f'{path}/train2014'):
        print("Downloading MSCOCO 2014 dataset...")
        wget.download(file_links['MSCOCO2014'], f'{path}/train2014.zip', bar=wget.bar_adaptive)
        with zipfile.ZipFile(f'{path}/train2014.zip',"r") as zip_ref:
            zip_ref.extractall(f"{path}")
        os.remove(f'{path}/train2014.zip')

    # Download the rices indexes
    if not os.path.exists(f'{path}/indexes'):
        print("Downloading rices indexes...")
        gdown.download(file_links['rices_indexes'], f'{path}/indexes.zip', quiet=False)
        with zipfile.ZipFile(f'{path}/indexes.zip',"r") as zip_ref:
            zip_ref.extractall(f"{path}")
        os.remove(f'{path}/indexes.zip')

    # Download the rices features
    if not os.path.exists(f'{path}/RICES-features'):
        os.mkdir(f'{path}/RICES-features')
        print("Downloading rices features...")
        gdown.download(file_links['rices_features'], f'{path}/RICES-features/coco.pkl', quiet=False)

def make_split():
    if not os.path.exists(f'{path}/prompt_train2014'):
        os.mkdir(f'{path}/prompt_train2014')
        prompt_karpathy_coco = json.load(open(f'{path}/prompt_karpathy_coco.json'))
        prompt_train_fileName = []

        for img in prompt_karpathy_coco['images']:
            if img['split'] == 'test':
                prompt_train_fileName.append(img['filename'])

        # Transfer to new folder
        for fileName in prompt_train_fileName:
            os.rename(f'{path}/train2014/{fileName}', f'{path}/prompt_train2014/{fileName}')

def make_dataset():
    print("Downloading files...")
    download_files()
    print("Making split...")
    make_split()
    print("Done!")

def download_checkpoint(model_name_or_path):
    if not os.path.exists(f'{path}/{model_name_or_path}.pt'):
        print("Downloading checkpoint...")
        hf_hub_download(f"openflamingo/{model_name_or_path}", "checkpoint.pt", local_dir=path)
        os.rename(f'{path}/checkpoint.pt', f'{path}/{model_name_or_path}.pt')

def update_path(model_name_or_path):
    #Update path to abs 
    scorer_params = json.load(open(f'{root}/config/scorer_params.json'))
    scorer_params['checkpoint_path'] = os.path.join(path, f'{model_name_or_path}.pt')
    scorer_params['coco_train_image_dir_path'] = f"{path}/train2014"
    scorer_params["coco_val_image_dir_path"] = f"{path}/prompt_train2014"
    scorer_params["coco_karpathy_json_path"] = f"{path}/prompt_karpathy_coco.json"
    scorer_params["coco_annotations_json_path"] = f"{path}/captions_train2014.json"
    scorer_params["cached_demonstration_features"] = f"{path}/RICES-features"
    json.dump(scorer_params, open(f'{root}/config/scorer_params.json', 'w'), indent=4)

def update_scorer_args(args):
    params = json.load(open(f'{root}/config/scorer_params.json', 'r'))
    params['shots'] = args.shots
    params['num_trials'] = args.num_trials
    params['cross_attn_every_n_layers'] = args.cross_attn_every_n_layers
    params['rices'] = args.rices
    params['lm_tokenizer_path'] = args.lm_tokenizer_path
    params['lm_path'] = args.lm_path
    params['is_distributed'] = args.is_distributed
    params['num_samples'] = args.num_samples
    json.dump(params, open(f'{root}/config/scorer_params.json', 'w'), indent=4)

def rices_setup():
    indice_folder = f'{path}/indexes'
    images_path = f'{path}/prompt_train2014'
    data_dir = indice_folder + "/metadata/metadata_0.parquet"
    df = pd.read_parquet(data_dir)
    df['image_path'] = df['image_path'].apply(lambda row: row.replace("/content/prompt_train2014", images_path))
    df.to_parquet(data_dir)


