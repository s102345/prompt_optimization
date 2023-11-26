export PYTHONPATH="../..:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5, 6, 7"

python ../manager.py \
--output_dir ./ \
--detailed_log \
--checkpoint_per_step \
--num_processes 8 \
--shots 2 \
--num_samples 200 \
--steps 150 \
--initial_prompt "What does the image describe?" \
--optimization_task_number 1 \
--example_number 3 \
--example_rule "rices" 
