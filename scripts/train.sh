export PYTHONPATH="../..:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

python manager.py \
--output_dir ./ \
--shots 2 \
--num_samples 200 \
--steps 5 \
--example_rule "rand" \
--initial_prompt "Output: " \
--extra_information \

