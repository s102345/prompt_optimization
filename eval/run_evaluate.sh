export PYTHONPATH="../..:$PYTHONPATH"

python ./evaluate_origin.py \
--coco_train_image_dir_path="../data/train2014" \
--coco_val_image_dir_path="../data/val2014" \
--coco_karpathy_json_path="../data/karpathy_coco_for_eval.json" \
--coco_annotations_json_path="../data/captions_val2014.json" \
--model=otter \
--model_path="../checkpoints/OTTER-Image-MPT7B" \
--checkpoint_path="../checkpoints/OTTER-Image-MPT7B/final_weights.pt" \
--device_map=auto \
--precision=fp16 \
--batch_size=8 \
--eval_coco \
--shots=2 \
--device=cuda \
--prompt_query="An image of" \
--prompt_example="An image of" \