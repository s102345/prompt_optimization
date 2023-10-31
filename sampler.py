from clip_filter import clip_filter
from appdata import root
import os, json, random, shutil, time

class Sampler():
    def __init__(self, seed=None):
        if seed != None:
            random.seed(seed)
        else:
            random.seed(time.time())
        self.init_used_images()
        
    def init_used_images(self):
        img_record = {}
        for img in os.listdir(f'{root}/data/prompt_train2014'):
            img_record[img] = False
        json.dump(img_record, open(f'{root}/tmp/used_images.json', 'w'), indent=4)

    def sample_image(self):
        used_images = json.loads(open(f'{root}/tmp/used_images.json', 'r').read())
        dataset = [img for img in used_images.keys() if not used_images[img]]
        if len(dataset) == 0:
            self.init_used_images()
            dataset = [img for img in used_images.keys() if not used_images[img]]
        image = random.choice(dataset)
        return image

    def rices_image(self, query, example_number=3):
        if os.path.exists(f'{root}/tmp/rices'):
            shutil.rmtree(f'{root}/tmp/rices')
        os.mkdir(f'{root}/tmp/rices')
        
        clip_filter(query, f'{root}/tmp/rices', f'{root}/data/indexes', num_results=example_number, threshold=None)
        
        result = os.listdir(f'{root}/tmp/rices')
        # Remove query image
        if query.split('/')[-1] in result:
            result.remove(query.split('/')[-1])
        else:
            result = result[:2]
        return result

    def update_record(self, used_images: list):
        used_record = json.load(open(f'{root}/tmp/used_images.json', 'r'))
        for img in used_images:
            used_record[img] = True
        json.dump(used_record, open(f'{root}/tmp/used_images.json', 'w'), indent=4)

    def search_image_info(self, image_name):
        annotations = json.load(open(f'{root}/data/prompt_karpathy_coco.json', 'r'))
        instances = json.load(open(f'{root}/data/instances_train2014.json', 'r'))

        target_info = dict()
        target_cat_id = []

        # Search image's info
        for info in annotations['images']:
            if info['filename'] == image_name:
                target_info = info
                break

        # Search image's categories id
        for info in instances['annotations']:
            if info['image_id'] == target_info['cocoid']:
                target_cat_id.append(info['category_id'])
    
        # Fetch target's caption
        target_cap = [sentence['raw'] for sentence in target_info['sentences']]

        # Translate categories id to categories name
        cat_dict = {cat['id']: cat['name'] for cat in instances['categories']}

        target_cat = {}
        target_cat_tmp = []
        for cat_id in target_cat_id:
            target_cat_tmp.append(cat_dict[cat_id])
        
        # Count
        for cat in list(set(target_cat_tmp)):
            target_cat[cat] = target_cat_tmp.count(cat)
        
        return {'Name': target_info['filename'], 'Captions': target_cap, 'Categories': target_cat}