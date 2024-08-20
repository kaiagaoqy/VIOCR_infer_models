from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import argparse
import torch
import json
from PIL import Image
import os
import tqdm
from transformers import TextStreamer
from utils import format_multi_choice
from collections.abc import Sequence
import re


def eval_model(tokenizer, model, image_processor, image_file, query):
    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles

    image = Image.open(image_file).convert('RGB')
    max_edge = max(image.size)  # resize to squared image for BEST performance.
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = DEFAULT_IMAGE_TOKEN + query
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
        model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    temperature = 0.7
    max_new_tokens = 512

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:])
    return outputs



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str,default='data/viocr/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/mplug.json')
    parser.add_argument('--img_dir', type=str,default='data/viocr/selected_images_new',help="Directory storing images")
    parser.add_argument('--model_path', type=str, default="MAGAer13/mplug-owl2-llama2-7b")
    parser.add_argument('--filter', nargs='+',default=["1","2","3","4","5","6","7","32","33","34","35","36","38","39","40","41"], help="low vision filter id")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = args.model_path
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device=device)


    samples = json.load(open(args.infile, "r"))
    output_samples = []
    q = "What are all the English words visible in the image?"


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
                
            output = eval_model(tokenizer, model, image_processor, image_file, q)
            output = output.strip()
            formatted_sample["rec_texts"] = output
            output_samples.append(formatted_sample)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True) 
    json.dump(output_samples, open(args.outfile, "w"), indent=4)
            



