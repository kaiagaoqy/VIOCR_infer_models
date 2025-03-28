import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
from torch.utils.data import DataLoader
from mmocr.apis import MMOCRInferencer
from tqdm import tqdm
import os
import json
import numpy as np
from collections import Counter
import pandas as pd
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
multi_li = [181,
276,
277,
166,
18,
146,
114,
210,
268,
117,
61,
95,
2,
245,
299,
24,
184,
99,
205,
264]

class CocoDetection(torchvision.datasets.CocoDetection):
    """

    Args:
        torchvision (_type_): _description_
    """
    def __init__(self, img_folder, ann_file, dataset_name,filter_id):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.dataset_name = dataset_name
        self.filter_id = filter_id

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        image_id = self.ids[idx]
        image_id = torch.tensor([image_id])
        # print("image_id: ",image_id,"length of target",len(target))
        target = {'image_id': image_id, 'annotations': target}
        target['dataset_name'] = self.dataset_name
        anno = target["annotations"]
        
        target = {}
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        ids = torch.tensor([obj["id"] for obj in anno], dtype=torch.int64)
        
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target["filter"] = self.filter_id
        target['dataset_name'] = self.dataset_name
        target['id'] = ids


        # for conversion to coco api
        area = [obj["area"] for obj in anno]
        iscrowd = [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        w, h = img.size
        target["orig_size"] =torch.tensor([int(h), int(w)])
        return F.to_tensor(img),target ## Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators
    
    
class collate_fn(object):
    def __init__(self):
        pass
    
    def __call__(self, batch):
        batch = list(zip(*batch))
        samples = batch[0]
        targets = batch[1]
        return samples, targets
    
    

# model_name = f'dbT_maerecB_{dataset_name}'
# infer = MMOCRInferencer(det='mmocr-dev-1.x/configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_totaltext.py',
#                         det_weights='mmocr-dev-1.x/checkpoint/dbnet_resnet18_fpnc_1200e_totaltext.pth',
#                         rec='mmocr-dev-1.x/configs/textrecog/maerec/maerec_b_union14m.py',
#                         rec_weights = 'mmocr-dev-1.x/checkpoint/maerec_b_union14m.pth')

# output_dir = './rec/maerec'

def main(model_name,BASEPATH,infer):
    
    for file_path in tqdm(glob.glob(BASEPATH+'*')):
        filter = int(file_path.split('/')[-1])
        print("Processing filter: ",filter)
        output_dir = 'results/totaltext_new/{}/'.format(filter)
        dataset = CocoDetection(img_folder=BASEPATH+str(filter),
                            ann_file='./viocr/data/viocr/anno.json',
                            dataset_name='totaltext',
                            filter_id=filter)

        sampler_val = torch.utils.data.SequentialSampler(dataset) if not dataset is None else None
        val_loader = DataLoader(dataset, batch_size=1,sampler=sampler_val,drop_last=False,collate_fn=collate_fn())
        
        results = []
        err = []
        batch = 1
        for imgs, targets in val_loader:
            img = imgs[0]
            img_arr = (img.permute(1,2,0).squeeze().numpy()*255).astype(np.uint8)
            try:
                preds = infer(inputs=img_arr, out_dir='outputs', save_pred=True)
                target = targets[0]
            except Exception as e:
                print(target['image_id'].to(device).item())
                err.append('id:{},image_id{}'.format(target['annotations'][0]['id'].to(device).item(),
                                                    target['annotations'][0]['image_id'].to(device).item()))
                continue
            image_id = target['image_id'].to(device).item()
            preds = preds['predictions'][0]
            for det_polygons,rec_texts,det_scores,rec_scores in zip(preds['det_polygons'],preds['rec_texts'],preds['det_scores'],preds['rec_scores']):
                result = {
                            'image_id': image_id,
                            # 'id':target['annotations'][0]['id'].to('cpu'),
                            'category_id': 1,
                            'polys': det_polygons,
                            'rec_texts': rec_texts,
                            # 'score': split_value.mean().item(),
                            # 'value': split_value.numpy().tolist(),
                            'det_score':det_scores,
                            'rec_score': rec_scores,
                            'filter': filter
                        }
                results.append(result)

        json_path = os.path.join(output_dir, model_name+'.json')
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(results,f)   
            
if __name__ == '__main__':
    dataset_name = 'totaltext'
    model_name = f'dbpp_maerecS_{dataset_name}'
    infer = MMOCRInferencer(det="dbnetpp",rec='maerec')
                        #rec='mmocr-dev-1.x/configs/textrecog/maerec/maerec_s_union14m.py',
                        #rec_weights = 'mmocr-dev-1.x/checkpoint/maerec_s_union14m.pth')
    BASEPATH = 'data/viocr/selected_images_new/'
    main(model_name,BASEPATH,infer)
