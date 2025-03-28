import torch
import os
import argparse
import json
import tqdm
from PIL import Image
from collections.abc import Sequence
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from dotenv import load_dotenv
load_dotenv(".env")
torch.manual_seed(1234)

device = "cuda" if torch.cuda.is_available() else "cpu"

def eval_model(tokenizer, model, image_file, query):
    query = tokenizer.from_list_format([
        {'image': image_file}, # Either a local path or an url
        {'text': query},
    ])
    output, history = model.chat(tokenizer, query=query, history=None)
    return output




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str,default='data/viocr/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/qwen.json')
    parser.add_argument('--img_dir', type=str,default='data/viocr/selected_images_new',help="Directory storing images")
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen-VL-Chat")
    parser.add_argument('--filter', nargs='+',default=["1","2","3","4","5","6","7","32","33","34","35","36","38","39","40","41"], help="low vision filter id")
    
    args = parser.parse_args()
    model_name = 'qwen'
    concept_file = args.infile.split('/')[-1]
    output_file = os.path.join(args.outfile,f'{concept_file}')
    
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # min_pixels = 5*28*28

    # use bf16
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", trust_remote_code=True, bf16=True).eval()
    # use fp16
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", trust_remote_code=True, fp16=True).eval()
    # use cpu only
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cpu", trust_remote_code=True).eval()
    # use cuda device
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda", trust_remote_code=True).eval()

    # Specify hyperparameters for generation
    # model.generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)
    
    samples = json.load(open(args.infile, "r"))['images']
    output_samples = []
    q = os.getenv("Prompt")

    for sample in tqdm.tqdm(samples):
        for filter_id in args.filter:

            formatted_sample = {"image_id":int(sample["id"]),
                                "category_id": 1,
                                "polys":[],
                                "rec_texts":"",
                                "rec_score":0,
                                "det_score":0,
                                "filter":int(filter_id) if int(filter_id) > 0 else int(sample["Filter_no"]),
                                }

            
            image_file = os.path.join(args.img_dir,filter_id, sample["file_name"]) if int(filter_id) > 0 else os.path.join(args.img_dir, sample["file_name"])
            if not os.path.exists(image_file):
                print(f"Image not found: {image_file}")
                continue
            output = eval_model(tokenizer, model, image_file, q)
            output = output.strip()
            formatted_sample['rec_texts'] = output
            output_samples.append(formatted_sample)
    
        
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    json.dump(output_samples, open(args.outfile, "w"), indent=4)
            



