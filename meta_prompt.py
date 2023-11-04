import os
import json
import sys
import random, time
from appdata import root
from utils.sampler import Sampler

class MetaPromptGenerator():
    def __init__(self, args, score_pair=None):
        self.args = args
        if self.args.seed != None:
            random.seed(args.seed)
        else:
            random.seed(time.time())
        self.sampler = Sampler(self.args.seed)

        if not os.path.exists(f'{root}/tmp'):
            os.mkdir(f'{root}/tmp')
            
        if self.args.last_step == 0:
            json.dump([], open(f'{root}/tmp/old_pair.json', 'w'), indent=4) # Tmp of old pairs
            json.dump([], open(f'{root}/tmp/all_prompt.json', 'w'), indent=4) # Tmp of all prompts
            default_meta_prompt = {
                "step" : 0,
                "meta-instruction": [
                    "I have provided several prompts, each paired with a score. These prompts are arranged in ascending order based on their scores. A higher score indicates better quality.",
                    "Below are some problems.",
                    "Please write a new prompt that is distinct from the ones provided. Aim for a score that's as high as possible.  The prompt should be concise, effective, and generally applicable to all problems above. Do not generate image captions. When submitting your text, encase it in square brackets. Do not mentions <INS> token. Prompt should be ended with preposition or colon."
                ],
                "solution-score pair": [],
                "optimization task": []
            }
            json.dump(default_meta_prompt, open(f'{root}/tmp/meta_prompt.json', 'w'), indent=4) # Tmp of meta-prompt
            # Init meta-prompt
            self.update_meta_prompt(score_pair)

        print("Meta-prompt initialized!")
    
    def update_score_pair(self, pair: list):
        # Read old prompt
        old_pair = json.load(open(f'{root}/tmp/old_pair.json', 'r'))
        # Merge new pair
        old_pair.extend(pair)   
        pair_dict = {}
        pair_count = {}
        # Count showup times(for average)
        for p in old_pair:
            if p['Prompt'] not in pair_count:
                pair_count[p['Prompt']] = 1
            else:
                pair_count[p['Prompt']] += 1

        for p in old_pair:
            if p['Prompt'] not in pair_dict:
                pair_dict[p['Prompt']] = p['Score'] / pair_count[p['Prompt']]
            else:
                pair_dict[p['Prompt']] = pair_dict[p['Prompt']] + p['Score'] / pair_count[p['Prompt']]
        
        new_pair = []
        record = set()
        for key, score in pair_dict.items():
            if key not in record:
                new_pair.append({'Prompt': key, 'Score': round(score, 2)})
                record.add(key)

        # Save pairs
        sorted_pair = sorted(new_pair, key=lambda x: x['Score'])
        json.dump(sorted_pair, open(f'{root}/tmp/old_pair.json', 'w'), indent=4)

        # Update meta-prompt
        prompt_file = json.load(open(f'{root}/tmp/meta_prompt.json', 'r')) 
        start_point = len(sorted_pair) - self.args.maximum_prompt_score_pair
        top_pair = sorted_pair[start_point if start_point > 0 else 0:]
        prompt_file['solution-score pair'] = top_pair
        json.dump(prompt_file, open(f'{root}/tmp/meta_prompt.json', 'w'), indent=4) 

        # Save all prompts
        all_prompt = json.load(open(f'{root}/tmp/all_prompt.json', 'r'))
        all_prompt.append(pair)
        json.dump(all_prompt, open(f'{root}/tmp/all_prompt.json', 'w'), indent=4)

    def update_optimization_task(self):
        old_prompt = json.load(open(f'{root}/tmp/meta_prompt.json', 'r'))

        # Fetch info
        for i in range(self.args.optimization_task_number):
            tmp = []
            task_examples = []
            target_img = self.sampler.sample_image()
            if self.args.example_rule == "rices":
                tmp.extend(self.sampler.rices_image(f"{root}/data/train2014/{target_img}", self.args.example_number))
            else:
                for i in range(self.args.example_number - 1):
                    tmp.append(self.sampler.sample_image())
            tmp.append(target_img)
            # Update used images
            used_record = []
            for img in tmp: 
                img_info = self.sampler.search_image_info(img)
                task_examples.append({
                    'image': img_info['Name'],
                    'captions': img_info['Captions'],
                    'extra_info': img_info['Categories']
                })
                used_record.append(img_info['Name'])
            self.sampler.update_record(used_record)
            # Save task
            # Update meta-prompt
            old_prompt['optimization task'].append(task_examples)
        
        json.dump(old_prompt, open(f'{root}/tmp/meta_prompt.json', 'w'), indent=4)

    def update_meta_prompt(self, score_pair):
        self.update_score_pair(score_pair)
        self.update_optimization_task()
        meta_prompt_info = json.load(open(f'{root}/tmp/meta_prompt.json', 'r'))
        meta_prompt_info['step'] += 1
        json.dump(meta_prompt_info, open(f'{root}/tmp/meta_prompt.json', 'w'), indent=4)

    def generate_meta_prompt(self):
        meta_prompt_info = json.load(open(f'{root}/tmp/meta_prompt.json', 'r'))
        meta_prompt = ""

        # Comporse meta-prompt
        meta_prompt += meta_prompt_info["meta-instruction"][0]
        meta_prompt += '\n\n'

        # Prompt-Score pair
        for pair in meta_prompt_info['solution-score pair']:
            meta_prompt += f"prompt: {pair['Prompt']}, score: {pair['Score']}\n"

        meta_prompt += '\n'
        meta_prompt += meta_prompt_info["meta-instruction"][1]
        meta_prompt += '\n\n'

        # Optimization task
        for i, task in enumerate(meta_prompt_info['optimization task']):
            meta_prompt += "Problem:\ninput: \n"
            # Example in task
            for j, task_examples in enumerate(task):
                if self.args.extra_information:
                    meta_prompt += "<img> {additional information: "
                    for info in task_examples['extra_info']:
                        amount = task_examples['extra_info'][info]
                        if info == list(task_examples['extra_info'])[-1]:
                            meta_prompt += f"{amount} {info}"
                        else:
                            meta_prompt += f"{amount} {info}, "
                    meta_prompt += "}\n"
                else:
                    meta_prompt += f"<img>\n"

                choosed_caption = random.choice(task_examples['captions'])

                if j == len(task) - 1:
                    meta_prompt += "<INS>\n"
                    meta_prompt += f"output: {choosed_caption}\n"
                else:
                    meta_prompt += f"<INS> {choosed_caption} <|endofchunk|>\n"

        meta_prompt += '\n'
        meta_prompt += meta_prompt_info["meta-instruction"][2]
        print(meta_prompt)
        return meta_prompt

    def get_top_pairs(self):
        meta_prompt = json.load(open(f'{root}/tmp/meta_prompt.json', 'r'))
        return meta_prompt['solution-score pair'] 
