export PYTHONPATH="../..:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

python ./prompt_optimization/manager.py \
--output_dir ./ \
--steps 5 \
--rices \
--shots 0 \
--example_rule "rand" \
--initial_prompt "Output: " \
--extra_information \

