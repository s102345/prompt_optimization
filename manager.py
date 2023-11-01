import json
import argparse


import os
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '29500'

from scorer import Scorer
import utils.prompt_utils as prompt_utils 
from meta_prompt import MetaPromptGenerator
from optimizer import Optimizer
from utils.appdata import root, path
import wandb
from statistics import mean

def get_args():
    parser = argparse.ArgumentParser(description='OpenFlamingo Prompt Optimization')

    # General parameters
    parser.add_argument('--output_dir', type=str, default="./", help='Output directory')
    parser.add_argument('--precision', type=str, default="fp16", help='Precision of model')
    parser.add_argument('--seed', default=None, type=int, help='Random seed')
    parser.add_argument('--detailed_log', type=int, default=-1, help='Output detailed prompt or not')

    # Model parameters
    parser.add_argument('--rices', action='store_true', help='Use rices to evaluate score or not')
    parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to evaluate on. -1 for all samples.")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run for each shot using different demonstrations")

    # Training parameters
    parser.add_argument('--steps', type=int, default=200, help='Number of steps')
    parser.add_argument('--instruction_per_step', type=int, default=8, help='Instructions generated per step')
    parser.add_argument('--initial_prompt', type=str, default="Output", help='Initial prompt')

    # Meta-prompt parameters
    parser.add_argument('--optimization_task_number', type=int, default=3, help='Amount of optimization tasks in meta prompt')
    parser.add_argument('--example_number', type=int, default=3, help='Example amount in each optimization task')
    parser.add_argument('--maximum_prompt_score_pair', type=int, default=20, help='Maximum number of prompt-score pair in meta prompt')
    parser.add_argument('--example_rule', type=str, default="rices", help='The way of choosing other examples in each optimization task')
    parser.add_argument('--extra_information', default=True, action="store_true", help='Extra information of image in meta prompt')

    return parser.parse_args()

class Manager():
    def __init__(self, args):
        self.args = args
        
        print("Initializing...")
        prompt_utils.make_dataset()
        prompt_utils.update_path()
        prompt_utils.update_scorer_args(self.args)
        prompt_utils.rices_setup()

        self.scorer = Scorer()
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

        print("Evaluating initial prompt...")
        initial_score = self.scorer.evaluate(args.initial_prompt)[0]
        self.metaPromptGenerator = MetaPromptGenerator(self.args, self.make_prompt_score_pair([self.args.initial_prompt], [initial_score])) 
        wandb.log({"CIDEr": initial_score})

    def make_prompt_score_pair(self, solutions, scores):
        prompt_score_pair = []
        for sol, score in zip(solutions, scores):
            prompt_score_pair.append({'Prompt': sol, 'Score': score})
        return prompt_score_pair
    
    def train(self):
        for i in range(1, self.args.steps + 1):
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
                score = self.scorer.evaluate(sol)
                scores.append(score)
                
            prompt_score_pair = self.make_prompt_score_pair(solutions, scores)
            self.metaPromptGenerator.update_meta_prompt(prompt_score_pair)

            # Log
            wandb.log({"CIDEr": mean(scores)})
            self.update_prompt_log()

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
    manager = Manager(args)
    manager.train()
    # End training
    top_pairs = manager.metaPromptGenerator.get_top_pairs()
    json.dump(top_pairs, open(f'{args.output_dir}/top_pairs.json', 'w'), indent=4)
    print("Done!")

if __name__ == '__main__':
    main()