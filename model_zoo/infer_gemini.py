
from dotenv import load_dotenv
from retry import retry
load_dotenv(".env")
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
import google.generativeai as genai
genai.configure(api_key=os.environ.get("Gemini_API_KEY"))


@retry((Exception), tries=3, delay=0, backoff=0)
def call_gemini(messages, model_name="models/gemini-1.5-flash", parse_fn=None):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(messages)
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
    parser.add_argument('--infile', type=str,default='data/viocr/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/gemini.json')
    parser.add_argument('--img_dir', type=str,default='data/viocr/selected_images_new',help="Directory storing images")
    parser.add_argument('--model_path', type=str, default="models/gemini-1.5-flash")
    parser.add_argument('--use_placeholder', action='store_true',help="Need to self-define placeholder in the question")
    parser.add_argument('--filter', nargs='+',default=["1","2","3","4","5","6","7","32","33","34","35","36","38","39","40","41"], help="low vision filter id")
    # choices=[
    #     "liuhaotian/llava-v1.6-34b", "liuhaotian/llava-v1.5-13b",
    #     "liuhaotian/llava-v1.5-7b", "liuhaotian/llava-v1.6-mistral-7b",
    #     "liuhaotian/llava-v1.6-vicuna-13b", "liuhaotian/llava-v1.6-vicuna-7b"
    #                         ]
    
    args = parser.parse_args()

    # format:
    # a list of dict
    # minimum keys: image, question
    # [
    #     {
    #         "image": "xxxx",
    #         "question": "xxxxx"},
    #     ...]
    # leave the output key empty
    
    samples = json.load(open(args.infile, "r"))
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
            image = Image.open(image_file)
            messages = [image, q]
            output = ""
            try:
                output = call_gemini(messages, model_name=args.model_path, parse_fn=lambda x: x.strip().replace(".", '').lower())
                formatted_sample["rec_texts"] = output
            except:
                print(f"Fail to call gemini for {sample['image'][0]}")
            
            formatted_samples.append(formatted_sample)
        
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True) 
    json.dump(formatted_samples, open(args.outfile, "w"), indent=4) 
            

