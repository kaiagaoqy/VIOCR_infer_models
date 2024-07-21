## https://huggingface.co/Salesforce/blip2-flan-t5-xl
import torch
import os
import argparse
import json
import tqdm
from PIL import Image
from collections.abc import Sequence
import re
# model
from transformers import Blip2Processor, Blip2ForConditionalGeneration


device = "cuda" if torch.cuda.is_available() else "cpu"


def eval_model(processor, model, image_file, query):
    raw_image = Image.open(image_file).convert("RGB")
    inputs = processor(raw_image, query, return_tensors="pt").to(device)
    # generated_ids = model.generate(**inputs,max_length=256)
    
    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        # num_beams=5,
        max_length=256,
        min_length=1,
    #     top_p=0.9,
    #     repetition_penalty=1.5,
    #     length_penalty=1.0,
    #     temperature=1,
    )
    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str,default='data/viocr/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/blip2_flan.json')
    parser.add_argument('--img_dir', type=str,default='data/viocr/selected_images_new',help="Directory storing images")
    parser.add_argument('--model_path', type=str, default="Salesforce/blip2-flan-t5-xl")
    parser.add_argument('--use_placeholder', action='store_true',help="Need to self-define placeholder in the question")
    parser.add_argument('--filter', nargs='+',default=["1","2","3","4","5","6","7","32","33","34","35","36","38","39","40","41"], help="low vision filter id")

    args = parser.parse_args()
    
    
    processor = Blip2Processor.from_pretrained(args.model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_path)
    model.to(device)
    
    # format:
    # a list of dict
    # minimum keys: image, question
    # [
    #     {
    #         "image": "xxxx",
    #         "question": "xxxxx"},
    #     ...]
    # leave the output key empty
    
    q = "What are all the English words visible in the image?"

    samples = json.load(open(args.infile, "r"))['images']
    model_output = []

    for sample in tqdm.tqdm(samples):
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
            
            output = eval_model(processor, model, image_file, q)
            
            output = output.strip()
            formatted_sample["rec_texts"] = output
            model_output.append(formatted_sample)
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    json.dump(model_output, open(args.outfile, "w"), indent=4)
            



