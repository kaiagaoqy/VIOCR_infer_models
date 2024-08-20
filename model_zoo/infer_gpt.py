
from openai import OpenAI
from dotenv import load_dotenv
from retry import retry
load_dotenv(".env")

from PIL import Image
import argparse
import torch
import json
import requests
from PIL import Image
from io import BytesIO
import re
import os
import tqdm
from collections.abc import Sequence
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

    
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

@retry((Exception), tries=3, delay=0, backoff=0)
def call_chatgpt(messages, model_name="gpt-4o", parse_fn=None):
    completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                # temperature=1,
                # max_tokens=256,
                # top_p=1,
                # frequency_penalty=0,
                # presence_penalty=0
            )
    if parse_fn is not None:
        ret = parse_fn(completion.choices[0].message.content)
    else:
        ret = completion.choices[0].message.content
    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str,default='data/viocr/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/gpt4o.json')
    parser.add_argument('--img_dir', type=str,default='data/viocr/selected_images_new',help="Directory storing images")
    parser.add_argument('--model_path', type=str, default="gpt-4o")
    parser.add_argument('--use_placeholder', action='store_true',help="Need to self-define placeholder in the question")
    parser.add_argument('--filter', nargs='+',default=["1","2","3","4","5","6","7","32","33","34","35","36","38","39","40","41"], help="low vision filter id")
    args = parser.parse_args()
    model_name = 'gpt-4o'


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
            messages = [{"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": [
                                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image}"
                                    }
                                },
                                                {
                                    "type": "text",
                                    "text": q
                                }
                            ]},]
            
            output = ""
            try:
                output = call_chatgpt(messages, model_name=args.model_path, parse_fn=lambda x: x.strip().replace(".", '').lower())
            except:
                print(f"fail on {sample['image'][0]}")
            formatted_sample["rec_texts"] = output
            formatted_samples.append(formatted_sample)
        
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True) 
    json.dump(formatted_samples, open(args.outfile, "w"), indent=4) 
            

