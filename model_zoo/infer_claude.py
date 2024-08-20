
from openai import OpenAI
from dotenv import load_dotenv
from retry import retry
load_dotenv(".env")

from PIL import Image
import argparse
import torch
import json
import requests
from io import BytesIO
import re
import os
import tqdm
from collections.abc import Sequence
import base64
import numpy as np
import anthropic

client = anthropic.Anthropic()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        img_byte = image_file.read()
        img_size = img_byte.__sizeof__()
        if img_size >= 5242880:
            img = Image.open(image_path).convert('RGB')
            w,h = img.size
            ratio = 1568/np.max([w,h])
            img = img.resize((int(w*ratio),int(h*ratio)))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            
            img_byte = img_byte_arr.getvalue()
            
        return base64.b64encode(img_byte).decode('utf-8')
    
    
@retry((Exception), tries=3, delay=0, backoff=0)
def call_claude(messages, model_name="claude-3-5-sonnet-20240620", parse_fn=None):
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are a helpful AI assistant.",
        messages=messages
    )
    if parse_fn is not None:
        ret = parse_fn(response.content[0].text)
    else:
        ret = response.content[0].text
    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str,default='data/viocr/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/claude.json')
    parser.add_argument('--img_dir', type=str,default='data/viocr/selected_images_new',help="Directory storing images")
    parser.add_argument('--model_path', type=str, default="claude-3-5-sonnet-20240620")
    parser.add_argument('--use_placeholder', action='store_true',help="Need to self-define placeholder in the question")
    parser.add_argument('--filter', nargs='+',default=["1","2","3","4","5","6","7","32","33","34","35","36","38","39","40","41"], help="low vision filter id")
    args = parser.parse_args()

    
    samples = json.load(open(args.infile, "r"))['images']
    formatted_samples = []
    q = "What are all the English words visible in the image?"

    for i, sample in enumerate(tqdm.tqdm(samples)):
        for filter_id in args.filter:
            formatted_sample = {"image_id":int(sample["id"]),
                                "category_id": 1,
                                "polys":[],
                                "rec_texts":"",
                                "rec_score":0,
                                "det_score":0,
                                "filter":int(filter_id),
                                }


            image_file = os.path.join(args.img_dir,filter_id, sample["file_name"])
            image = encode_image(image_file)
            postfix = image_file.split(".")[-1].lower()
            if postfix == "png":
                media_type = "image/png"
            elif postfix == "jpg" or postfix == "jpeg":
                media_type = "image/jpeg"
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image,
                                },
                            },
                            {
                                "type": "text",
                                "text": q
                            }
                        ],
                    }
            ]
            try:
                output = call_claude(messages, model_name=args.model_path, parse_fn=lambda x: x.strip().replace(".", '').lower())
                formatted_sample["rec_texts"] = output
            except:
                print(f'fail on {image_file}')
                
            formatted_samples.append(formatted_sample)
        
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True) 
    json.dump(formatted_samples, open(args.outfile, "w"), indent=4) 
            

