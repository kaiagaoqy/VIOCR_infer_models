
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

resume_file = "resume_gpt.txt"
resume_index = 0
# If a resume file exists, read the index to resume from.
if os.path.exists(resume_file):
    try:
        with open(resume_file, "r") as f:
            resume_index = int(f.read().strip().split()[0])
        print(f"Resuming from sample index: {resume_index}")
    except Exception as e:
        print("Failed to read resume file, starting from index 0")

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
    parser.add_argument('--infile', type=str,default='data/etdrs/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/etdrs/gpt4o.json')
    parser.add_argument('--img_dir', type=str,default='data/etdrs',help="Directory storing images")
    parser.add_argument('--model_path', type=str, default="gpt-4o")
    parser.add_argument('--filter', nargs='+',default=["0"], help="low vision filter id")
    args = parser.parse_args()
    model_name = 'gpt-4o'
    should_exit = False


    samples = json.load(open(args.infile, "r"))['images']
    formatted_samples = []
    # q = "What are all the English words visible in the image? \
    #     Please output each word in order from left to right and top to bottom.\
    #     If you can not recognize it, please leave it blank. Please return strictly in the following format (do not output anything else):\
    #     ['word1', 'word2', 'word3', ...]\
    #     Note that the data such as 'word1' above should be replaced with real word."
    
    q = os.getenv("Prompt")
    for i, sample in enumerate(tqdm.tqdm(samples)):
        # Skip samples until we reach the resume index.
        if i < resume_index:
            continue
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
            image = encode_image(image_file)
            messages = [{"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": [
                                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image}"
                                    }
                                },
                                                {
                                    "type": "text",
                                    "text": q
                                }
                            ]},]
            
            output = ""
            try:
                output = call_chatgpt(messages, model_name=args.model_path, parse_fn=lambda x: x.strip().replace(".", ''))
            except Exception as e:
                print(f"Error: {e}")
                # Check if the error message contains 429
                if "429" in str(e):
                    print("Encountered error 429, saving resume index and exiting.")
                    with open(resume_file, "w") as f:
                        f.write(f'{str(i)} - {sample}')
                    should_exit = True
                    break  # Breaks out of inner loop
                else:
                    print(f"Fail on {sample['file_name']}")
            formatted_sample["rec_texts"] = output
            formatted_samples.append(formatted_sample)
        if should_exit:
            break
        
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True) 
    json.dump(formatted_samples, open(args.outfile, "w"), indent=4) 
    # Optionally, if processing completes successfully, remove the resume file.
    if not should_exit and os.path.exists(resume_file):
        os.remove(resume_file)
            

