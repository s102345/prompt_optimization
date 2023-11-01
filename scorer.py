import json
import os
import subprocess

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eval'))

from utils.appdata import root

class Scorer():
    def __init__(self):
        print("Scorer initialized!")

        os.environ["PYTHONPATH"] = "../..:$PYTHONPATH"
        os.environ["CUDA_VISIBLE_DEVICES"] = 3

        self.configs = json.load(open(f'{root}/config/scorer_params.json'))

    def evaluate(self, prompt):
        output = subprocess.run(["python -m eval.evaluate\\",
                        f"--coco_train_image_dir_path= {self.configs['coco_train_image_dir_path']} \\",
                        f"--coco_val_image_dir_path= {self.configs['coco_val_image_dir_path']} \\",
                        f"--coco_karpathy_json_path= {self.configs['coco_karpathy_json_path']} \\",
                        f"--coco_annotations_json_path= {self.configs['coco_annotations_json_path']} \\",
                        "--model=otter\\",
                        f"--model_path= {self.configs['model_path']} \\",
                        f"--checkpoint_path= {self.configs['checkpoint_path']} \\",
                        "--device_map=auto",
                        f"--precision= {self.configs['precision']} \\",
                        f"--batch_size= {self.configs['batch_size']} \\",
                        "--eval_coco"
                        f"--shots= {self.configs['shots']} \\",
                        f"--num_trials= {self.configs['num_trials']} \\",
                        f"--num_samples= {self.configs['num_samples']} \\",
                        f"--rices= {self.configs['rices']} \\",
                        f"--cached_demonstration_features= {self.configs['cached_demonstration_features']} \\",
                        f"--prompt= {prompt} \\",
                        ], capture_output=True, shell=True, text=True)
        print(output.stdout)
        return 0





