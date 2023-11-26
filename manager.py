import json
import argparse

import os
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '29500'

from scorer import Scorer
import utils.prompt_utils as prompt_utils 
from meta_prompt import MetaPromptGenerator
from optimizer import Optimizer
from appdata import root, path
import wandb
import shutil
from statistics import mean

import dill
import multiprocessing as mp

def get_args():
    parser = argparse.ArgumentParser(description='OpenFlamingo Prompt Optimization')

    # General parameters
    parser.add_argument('--output_dir', type=str, default="./", help='Output directory')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--detailed_log', action='store_true', help='Output detailed prompt or not')
    parser.add_argument('--checkpoint_per_step', action='store_true', help='Frequency of saving checkpoints')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to use for evaluation')

    # Scorer model parameters
    parser.add_argument('--precision', type=str, default="fp16", help='Precision of model')
    parser.add_argument('--rices', action='store_true', help='Use rices to evaluate score or not')
    parser.add_argument("--shots", nargs="+", default=2, type=int)
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to evaluate on. -1 for all samples.")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run for each shot using different demonstrations")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for scorer")

    # Training parameters
    parser.add_argument('--steps', type=int, default=5, help='Number of steps')
    parser.add_argument('--last_step', type=int, default=0, help='Last step')
    parser.add_argument('--instruction_per_step', type=int, default=8, help='Instructions generated per step')
    parser.add_argument('--initial_prompt', type=str, default="Output", help='Initial prompt')

    # Meta-prompt parameters
    parser.add_argument('--optimization_task_number', type=int, default=3, help='Amount of optimization tasks in meta prompt')
    parser.add_argument('--example_number', type=int, default=3, help='Example amount in each optimization task')
    parser.add_argument('--maximum_prompt_score_pair', type=int, default=20, help='Maximum number of prompt-score pair in meta prompt')
    parser.add_argument('--example_rule', type=str, default="rand", help='The way of choosing other examples in each optimization task')
    parser.add_argument('--extra_information', action="store_true", help='Extra information of image in meta prompt')

    return parser.parse_args()

class Manager():
    def __init__(self, args):
        self.args = args
        
        print("Initializing...")
        prompt_utils.make_dataset()
        prompt_utils.update_path()
        prompt_utils.rices_setup()

        self.scorer = Scorer(args)
        self.optimizer = Optimizer()

        #Log
        wandb.init(project="Optimization by PROmpting")
        config = {
            "scorer_rices": self.args.rices,
            "scorer_shots": self.args.shots,
            "scorer_num_trials": self.args.num_trials,
            "scorer_num_samples": self.args.num_samples,
            "steps": self.args.steps,
            "instruction_per_step": self.args.instruction_per_step,
            "initial_prompt": self.args.initial_prompt,
            "optimization_task_number": self.args.optimization_task_number,
            "example_number": self.args.example_number,
            "maximum_prompt_score_pair": self.args.maximum_prompt_score_pair,
            "example_rule": self.args.example_rule,
            "extra_information": self.args.extra_information,
        }
        wandb.config.update(config)
        if self.args.last_step == 0:
            print("Evaluating initial prompt...")
            self.scorer.do_sample()
            initial_score = self.scorer.evaluate(args.initial_prompt, -1)
            print(f"Initial score: {initial_score}")
            self.metaPromptGenerator = MetaPromptGenerator(self.args, self.make_prompt_score_pair([self.args.initial_prompt], [initial_score])) 
            wandb.log({"CIDEr": initial_score}, step=0)
        else:
            print("Loading meta prompt...")
            self.metaPromptGenerator = MetaPromptGenerator(self.args)

    def make_prompt_score_pair(self, solutions, scores):
        prompt_score_pair = []
        for sol, score in zip(solutions, scores):
            prompt_score_pair.append({'Prompt': sol, 'Score': score})
        return prompt_score_pair
    
    def train(self):
        for i in range(self.args.last_step, self.args.steps):
            # LOOP
            # Receive meta-prompt
            meta_prompt = self.metaPromptGenerator.generate_meta_prompt()
            # Use meta-prompt to generate solutions
            solutions = []
            scores = []
            self.optimizer.init()
            
           
            for j in range(self.args.instruction_per_step):
                sol = self.optimizer.generate(meta_prompt)
                solutions.append(sol)

            self.scorer.do_sample()
            
            if self.args.num_processes == 1:
                for j in range(self.args.instruction_per_step):
                    score = self.scorer.evaluate(solutions[j], -1)
                    scores.append(score)

            else:
                for j in range(0, self.args.instruction_per_step, self.args.num_processes):
                    num_processes = min(self.args.num_processes, self.args.instruction_per_step - j)
                    return_queue = mp.Queue()
                    processes = []

                    for rank in range(num_processes):
                        prompt = solutions[j + rank]
                        p = mp.Process(target=self.scorer.evaluate, args=(prompt, rank, return_queue, ))
                        p.start()
                        processes.append(p)
                    
                    results = [return_queue.get() for _ in range(num_processes)]

                    for p in processes:
                        p.join()
                    
                    results.sort(key=lambda x: x['rank'])
                    results = [result['score'] for result in results]
                    scores.extend(results)
       
            prompt_score_pair = self.make_prompt_score_pair(solutions, scores)
            self.metaPromptGenerator.update_meta_prompt(prompt_score_pair)

            # Log
            wandb.log({"CIDEr": mean(scores)}, step=i)
            self.update_prompt_log()

            # Checkpoint
            if self.args.checkpoint_per_step:
                shutil.copy(f'{root}/tmp/meta_prompt.json', f'{self.args.output_dir}/meta_prompt.json')
                shutil.copy(f'{root}/tmp/used_images.json', f'{self.args.output_dir}/used_images.json')
                shutil.copy(f'{root}/tmp/old_pair.json', f'{self.args.output_dir}/old_pair.json')
                shutil.copy(f'{root}/tmp/all_prompt.json', f'{self.args.output_dir}/all_prompt.json')

    def update_prompt_log(self):
        all_prompt = json.load(open(f'{root}/tmp/all_prompt.json', 'r'))
        data = []
        for step, prompts in enumerate(all_prompt):
            for prompt in prompts:
                data.append([step, prompt['Score']])

        table = wandb.Table(data=data, columns = ["step", "score"])
        scatter = wandb.plot.scatter(table, "step", "score", title="Scatter Plot")
        wandb.log({"scatter_plot": scatter})
        
def main():
    args = get_args()
    mp.set_start_method('spawn')

    if not os.path.exists(f'{args.output_dir}'):
        os.mkdir(f'{args.output_dir}')

    manager = Manager(args)
    manager.train()
    # End training

    # Output files
    # Results of top_pairs
    top_pairs = manager.metaPromptGenerator.get_top_pairs()
    json.dump(top_pairs, open(f'{args.output_dir}/top_pairs.json', 'w'), indent=4)

    # Detailed log  
    if args.detailed_log:
        shutil.copy(f'{root}/tmp/all_prompt.json', f'{args.output_dir}/all_prompt.json')

    print("Done!")

if __name__ == '__main__':
    main()