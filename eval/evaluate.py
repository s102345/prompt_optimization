import argparse
import importlib
import json
import os
import random
import uuid
from collections import defaultdict

from einops import repeat
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .coco_metric import compute_cider, postprocess_captioning_generation
from .eval_datasets import CaptionDataset
from tqdm import tqdm

from .eval_model import BaseEvalModel
from .distributed import init_distributed_device, world_info_from_env

from .rices import RICES

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` and `Otter` is supported.",
    default="otter",
)
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    help="Huggingface format Otter or OpenFlamingo model.",
    default="/home/luodian/projects/checkpoints/flamingo-mpt-30B-pretrain-mix-bf16",
)
parser.add_argument("--results_file", type=str, default=None, help="JSON file to save results")

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument("--query_set_size", type=int, default=2048, help="Size of demonstration query set")

parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument(
    "--prompt",
    type=str,
    default="",
    help="Prompt use for evaluation",
)

parser.add_argument(
    "--rices",
    action="store_true",
    help="Whether to use RICES for evaluation. If False, uses random demonstrations.",
)
parser.add_argument(
    "--rices_vision_encoder_path",
    default="ViT-L-14",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--rices_vision_encoder_pretrained",
    default="openai",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--cached_demonstration_features",
    default=None,
    help="Directory where rices features for all choices of in-context examples are stored as a pkl file with the dataset name. If None, features are re-computed by script.",
)

# Per-dataset evaluation flags
parser.add_argument(
    "--eval_coco",
    action="store_true",
    default=False,
    help="Whether to evaluate on COCO.",
)
parser.add_argument(
    "--eval_flickr30",
    action="store_true",
    default=False,
    help="Whether to evaluate on Flickr30.",
)
# Dataset arguments

## COCO Dataset
parser.add_argument(
    "--coco_train_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_val_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_karpathy_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    default=None,
)

## Flickr30 Dataset
parser.add_argument(
    "--flickr_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None,
)
parser.add_argument(
    "--flickr_karpathy_json_path",
    type=str,
    help="Path to the dataset_flickr30k.json file.",
    default=None,
)
parser.add_argument(
    "--flickr_annotations_json_path",
    type=str,
    help="Path to the dataset_flickr30k_coco_style.json file.",
)

# Distributed evaluation
parser.add_argument(
    "--world_size",
    default=1,
    type=int,
    help="number of distributed processes",
)
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument(
    "--horovod",
    default=False,
    action="store_true",
    help="Use horovod for distributed training.",
)
parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
)

def main():
    args, leftovers = parser.parse_known_args()
    module = importlib.import_module(f"pipeline.eval.models.{args.model}")

    # print("======================================")
    # print(args)
    # print("======================================")
    # print(leftovers)
    # print("======================================")

    # set up distributed evaluation
    model_args = {leftover.lstrip("-").split("=")[0]: leftover.split("=")[1] for leftover in leftovers}
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)

    prompt = args.prompt

    eval_model = module.EvalModel(model_args)
    eval_model.set_device(device_id)
    if device_id != torch.device("cpu") and args.world_size > 1:
        eval_model.init_distributed()

    if args.model != "open_flamingo" and args.model != "otter" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    if args.eval_flickr30:
        print("Evaluating on Flickr30k...")

        # load cached demonstration features for RICES
        if args.cached_demonstration_features is not None:
            cached_features = torch.load(
                f"{args.cached_demonstration_features}/flickr30.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_captioning(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="flickr",
                    cached_features=cached_features,
                    min_generation_length=0,
                    max_generation_length=128,
                    num_beams=3,
                    prompt=prompt,
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                    scores.append(cider_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean CIDEr score: {np.nanmean(scores)}")
                results["flickr30"].append({"shots": shot, "trials": scores, "mean": np.nanmean(scores)})

        if args.rank == 0 and args.results_file is not None:
            with open(args.results_file, "w") as f:
                json.dump(results, f)

    if args.eval_coco:
        print("Evaluating on COCO...")

        # load cached demonstration features for RICES
        if args.cached_demonstration_features is not None:
            cached_features = torch.load(
                f"{args.cached_demonstration_features}/flickr30.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_captioning(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="coco",
                    cached_features=cached_features,
                    min_generation_length=0,
                    max_generation_length=64,
                    num_beams=3,
                    prompt=prompt,
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                    scores.append(cider_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean CIDEr score: {np.nanmean(scores)}")
                results["coco"].append({"shots": shot, "trials": scores, "mean": np.nanmean(scores)})
        if args.rank == 0 and args.results_file is not None:
            with open(args.results_file, "w") as f:
                json.dump(results, f)

    if args.rank == 0 and args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)

def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(f"num_samples + query_set_size must be less than {len(full_dataset)}")

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(len(full_dataset), num_samples + query_set_size, replace=False)
    return random_indices


def get_query_set(train_dataset, query_set_size, seed):
    np.random.seed(seed)
    query_set = np.random.choice(len(train_dataset), query_set_size, replace=False)
    return [train_dataset[i] for i in query_set]


def prepare_eval_samples(test_dataset, num_samples, batch_size, seed):
    np.random.seed(seed)
    random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    dataset = torch.utils.data.Subset(test_dataset, random_indices)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )
    return loader


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]


def compute_effective_num_shots(num_shots, model_type):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def evaluate_captioning(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 20,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    num_shots: int = 8,
    dataset_name: str = "coco",
    cached_features=None,
    prompt: str = "",
):
    """Evaluate a model on COCO dataset.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 20.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        dataset_name (str, optional): dataset to evaluate on. Can be "coco" or "flickr". Defaults to "coco".
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.
    Returns:
        float: CIDEr score

    """

    if dataset_name == "coco":
        image_train_dir_path = args.coco_train_image_dir_path
        image_val_dir_path = args.coco_val_image_dir_path
        annotations_path = args.coco_karpathy_json_path
    elif dataset_name == "flickr":
        image_train_dir_path = args.flickr_image_dir_path  # Note: calling this "train" for consistency with COCO but Flickr only has one split for images
        image_val_dir_path = None
        annotations_path = args.flickr_karpathy_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_dataset = CaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=True,
        dataset_name=dataset_name if dataset_name != "nocaps" else "coco",
    )

    test_dataset = CaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
        seed,
    )

    if args.rices:
        rices_dataset = RICES(
            train_dataset,
            eval_model.device,
            args.batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
    else:
        # subset of the training set to sample context images from
        in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)

    predictions = defaultdict()

    np.random.seed(seed + args.rank)  # make sure each worker has a different seed for the random context samples
    for batch in tqdm(
        test_dataloader,
        desc=f"Running inference {dataset_name.upper()}",
        disable=args.rank != 0,
    ):
        if args.rices:
            batch_demo_samples = rices_dataset.find(batch["image"], effective_num_shots)
        else:
            batch_demo_samples = sample_batch_demos_from_query_set(in_context_samples, effective_num_shots, len(batch["image"]))

        batch_images = []
        batch_text = []
        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch["image"][i]])

            context_text = "".join([eval_model.get_caption_prompt(prompt=prompt, caption=x["caption"].strip()) for x in batch_demo_samples[i]])

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(context_text + eval_model.get_caption_prompt(prompt=prompt))

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            min_generation_length=min_generation_length,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        new_predictions = [postprocess_captioning_generation(out).replace('"', "") for out in outputs]

        for i, sample_id in enumerate(batch["image_id"]):
            predictions[sample_id] = {
                "caption": new_predictions[i],
            }

    # all gather
    all_predictions = [None] * args.world_size
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of dicts

    if args.rank != 0:
        return

    all_predictions = {k: v for d in all_predictions for k, v in d.items()}  # merge dicts
    print(f"In total {len(all_predictions)} predictions.")

    # save the predictions to a temporary file
    results_path = f"{dataset_name}results_{uuid.uuid4()}.json"

    with open(results_path, "w") as f:
        f.write(
            json.dumps(
                [{"image_id": k, "caption": all_predictions[k]["caption"]} for k in all_predictions],
                indent=4,
            )
        )

    metrics = compute_cider(
        result_path=results_path,
        annotations_path=args.coco_annotations_json_path if dataset_name == "coco" else args.flickr_annotations_json_path,
    )

    # delete the temporary file
    os.remove(results_path)

    return metrics["CIDEr"] * 100.0

if __name__ == "__main__":
    main()
