import json
import os
import random
import importlib

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eval'))

from appdata import root
from eval.evaluate import main as evaluate_main

class Scorer():
    def __init__(self, args):
        print("Start scorer initializing!")
        # Load configs
        self.configs = json.load(open(f'{root}/config/scorer_params.json'))

        # Set args
        self.args = args
        self.update_args()
        self.update_config()

        # Set random seed
        random.seed(self.args.seed)

        # Setup model
        self.setup_model()
        print("Scorer initialized!")

    def setup_model(self):
        module = importlib.import_module(f"eval.models.{self.args.model}")
        self.eval_model = module.EvalModel(self.configs)

    def update_config(self):
        self.configs['model'] = self.args.model
        self.configs['device_map'] = self.args.device_map   
        self.configs['precision'] = self.args.precision
        self.configs['device'] = "cuda"

    def update_args(self):
        self.args.coco_train_image_dir_path = self.configs['coco_train_image_dir_path']
        self.args.coco_val_image_dir_path = self.configs['coco_val_image_dir_path']
        self.args.coco_karpathy_json_path = self.configs['coco_karpathy_json_path']
        self.args.coco_annotations_json_path = self.configs['coco_annotations_json_path']
        self.args.model = 'otter'
        self.args.model_path = self.configs['model_path']
        self.args.checkpoint_path = self.configs['checkpoint_path']
        self.args.cached_demonstration_features = self.configs['cached_demonstration_features']
        self.args.eval_coco = True
        self.args.device_map = 'auto'

        if type(self.args.shots) == int:
            self.args.shots = [self.args.shots]

    def do_sample(self):
        # We do evaluate on train2014
        # which the source of train2014 & val2014 are the same
        annotations = json.load(open(self.configs['coco_karpathy_json_path']))
        train_images = os.listdir(os.path.join(root, 'data', 'train2014'))
        
        if self.args.num_samples > len(train_images):
            print("num_samples is larger than the number of train images")
            exit(1)
        elif self.args.num_samples == -1:
            print("num_samples of -1 is not supported now")
            exit(1)

        sampled_images = random.sample(train_images, self.args.num_samples)
        unsampled_images = list(set(train_images) - set(sampled_images))

        for idx, ann in enumerate(annotations['images']):
            # unsampled_images -> train
            if ann['filename'] in unsampled_images:
                annotations['images'][idx]['split'] = 'train'
            # sampled_images -> test
            elif ann['filename'] in sampled_images:
                annotations['images'][idx]['split'] = 'test'
            # test -> val(something unused)
            else:
                annotations['images'][idx]['split'] = 'val'

        json.dump(annotations, open(self.configs['coco_karpathy_json_path'], 'w'))

    def evaluate(self, prompt):
        self.do_sample()
        self.args.prompt = prompt
        score = evaluate_main(self.args, self.eval_model)
        print("Score:", score)
        return score
    




