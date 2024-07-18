## https://huggingface.co/docs/transformers/main/model_doc/git#transformers.GitForCausalLM.forward.example
## Example: question = "what does the front of the bus say at the top?"
import torch
import os
import argparse
import json
import tqdm
from PIL import Image
from collections.abc import Sequence
import re
# model
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor


device = "cuda" if torch.cuda.is_available() else "cpu"


def eval_model(processor, model, image_file, query):
    raw_image = Image.open(image_file).convert("RGB")
    inputs = processor(images=raw_image, text=query, padding='longest', do_convert_rgb=True, return_tensors="pt").to(device)
    inputs = inputs.to(dtype=model.dtype)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=496)

    output = processor.decode(output[0], skip_special_tokens=True)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str,default='data/viocr/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/pali.json')
    parser.add_argument('--img_dir', type=str,default='data/viocr/selected_images_new',help="Directory storing images")
    parser.add_argument('--model_path', type=str, default="google/paligemma-3b-mix-224")
    parser.add_argument('--use_placeholder', action='store_true',help="Need to self-define placeholder in the question")
    parser.add_argument('--filter', nargs='+',default=["1","2","3","4","5","6","7","32","33","34","35","36","38","39","40","41"], help="low vision filter id")

# Use like:
# python arg.py --filter 1234 2345 3456 4567
    args = parser.parse_args()

    ## Need to accept license first https://huggingface.co/google/paligemma-3b-mix-224
    model = PaliGemmaForConditionalGeneration.from_pretrained(args.model_path)
    processor = PaliGemmaProcessor.from_pretrained(args.model_path)
    model.to(device)
    q = "What are all the English words visible in the image?"

    model.to(device)


    samples = json.load(open(args.infile, "r"))['images']
    
    model_output = []
    #  {
    #     "image_id": 2,
    #     "category_id": 1,
    #     "polys": [],
    #     "rec_texts": "",
    #     "rec_score": 0,
    #     "det_score": 0,
    #     "filter": 40
    #   }
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
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
            print(output)
            print(image_file.strip())
            output = output.strip()
            if re.search(r"\n|\.", output):
                output = re.split(r"\n|\.",output)[-1].strip()
                
            formatted_sample["rec_texts"] = output
            model_output.append(formatted_sample)
        
    
        json.dump(model_output, open(args.outfile, "w"), indent=4)
