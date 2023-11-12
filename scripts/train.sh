export PYTHONPATH="../..:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0"

python ../manager.py \
--output_dir ./ \
--detailed_log \
--checkpoint_per_step \
--num_processes 1 \
--shots 2 \
--num_samples 200 \
--steps 5 \
--initial_prompt "A Image of" \
--example_rule "rand" \
--extra_information \
