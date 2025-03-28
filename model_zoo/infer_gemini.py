
from dotenv import load_dotenv
from retry import retry

import os
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
from PIL import Image
from google import genai

load_dotenv(".env")
resume_file = "resume_gemini.txt"
resume_index = 0
# If a resume file exists, read the index to resume from.
if os.path.exists(resume_file):
    try:
        with open(resume_file, "r") as f:
            resume_index = int(f.read().strip().split()[0])
        print(f"Resuming from sample index: {resume_index}")
    except Exception as e:
        print("Failed to read resume file, starting from index 0")


client = genai.Client(api_key=os.environ.get("Gemini-API-KEY"))


@retry((Exception), tries=3, delay=0, backoff=0)
def call_gemini(messages, model_name="gemini-1.5-flash", parse_fn=None):
    response = client.models.generate_content(
        model=model_name,
        contents=messages,
        # temperature=0.7,
        # max_output_tokens=512,
        # top_p=0.95,
        # top_k=40,
        # stop_sequences=["\n"],
    )
    ret = ""
    try:
        if parse_fn is not None:
            ret = parse_fn(response.text)
        else:
            ret = response.text
    except:
        print(response.prompt_feedback)
    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str,default='data/mnread/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/gemini_15_flash.json')
    parser.add_argument('--img_dir', type=str,default='data/mnread',help="Directory storing images")
    parser.add_argument('--model_path', type=str, default="gemini-1.5-flash")
    parser.add_argument('--use_placeholder', action='store_true',help="Need to self-define placeholder in the question")
    parser.add_argument('--filter', nargs='+',default=["0"], help="low vision filter id")

    
    args = parser.parse_args()
    should_exit = False


    
    samples = json.load(open(args.infile, "r"))['images']
    formatted_samples = []
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
            image = Image.open(image_file)
            messages = [image, q]
            output = ""
            try:
                output = call_gemini(messages, model_name=args.model_path, parse_fn=lambda x: x.strip().replace(".", ''))
                formatted_sample["rec_texts"] = output
            except Exception as e:
                print(f"Error: {e}")
                should_exit = True
            formatted_sample["rec_texts"] = output
            formatted_samples.append(formatted_sample)
        if should_exit:
            break
            
        
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True) 
    json.dump(formatted_samples, open(args.outfile, "w"), indent=4) 
    # Optionally, if processing completes successfully, remove the resume file.
    if not should_exit and os.path.exists(resume_file):
        os.remove(resume_file)
            

            

