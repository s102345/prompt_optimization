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

from .eval_utils import compute_effective_num_shots, get_query_set, prepare_eval_samples, sample_batch_demos_from_query_set, merge_args
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
parser.add_argument("--prompt_example", type=str, default=None, help="Prompt to use for evaluation(part example)")
parser.add_argument("--prompt_query", type=str, default=None, help="Prompt to use for evaluation(part query)")
parser.add_argument("--batch_size", type=int, default=2)

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

def main():
    args, leftovers = parser.parse_known_args()
    module = importlib.import_module(f"eval.models.{args.model}")

    model_args = {leftover.lstrip("-").split("=")[0]: leftover.split("=")[1] for leftover in leftovers}
    eval_model = module.EvalModel(model_args)

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
                    num_beams=3
                )
                print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                scores.append(cider_score)

            print(f"Shots {shot} Mean CIDEr score: {np.nanmean(scores)}")
            results["flickr30"].append({"shots": shot, "trials": scores, "mean": np.nanmean(scores)})

    if args.eval_coco:
        print("Evaluating on COCO...")

        # load cached demonstration features for RICES
        if args.cached_demonstration_features is not None:
            cached_features = torch.load(
                f"{args.cached_demonstration_features}/coco.pkl", map_location="cpu"
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
                    num_beams=3
                )
                print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                scores.append(cider_score)

            print(f"Shots {shot} Mean CIDEr score: {np.nanmean(scores)}")
            results["coco"].append({"shots": shot, "trials": scores, "mean": np.nanmean(scores)})

    if args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)

    if args.eval_coco:
        return results['coco'][0]['mean']
    if args.eval_flickr30:
        return results[0]['flickr30']['mean']

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
    cached_features=None
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

    np.random.seed(seed)
    for batch in tqdm(
        test_dataloader,
        desc=f"Running inference {dataset_name.upper()}",
        #disable=args.rank != 0,
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

            if args.prompt_example is not None:
                context_text = "".join([eval_model.get_caption_prompt(prompt=args.prompt_example, caption=x["caption"].strip()) for x in batch_demo_samples[i]])
            else:
                context_text = "".join([eval_model.get_caption_prompt(caption=x["caption"].strip()) for x in batch_demo_samples[i]])


            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            if args.prompt_query is not None:
                batch_text.append(context_text + eval_model.get_caption_prompt(prompt=args.prompt_query))
            else:
                batch_text.append(context_text + eval_model.get_caption_prompt())

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

    all_predictions = predictions

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
