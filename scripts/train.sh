export PYTHONPATH="../..:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0,1"

python ../manager.py \
--output_dir ./ \
--shots 2 \
--num_processes 8 \
--num_samples 400 \
--rices \
--steps 10 \
--example_rule "rand" \
--initial_prompt "Output: " \
--extra_information \
