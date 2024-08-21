## Model Zoo https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

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

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(model_name, tokenizer, model, image_processor, context_len, args):
    # Model
    disable_torch_init()

    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     args.model_path, args.model_base, model_name
    # )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    return outputs


def load_model(model_path, model_base):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base, model_name
        )
    return model_name, tokenizer, model, image_processor, context_len

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str,default='data/viocr/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/llava.json')
    parser.add_argument('--img_dir', type=str,default='data/viocr/selected_images_new',help="Directory storing images")
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.6-34b")
    parser.add_argument('--use_placeholder', action='store_true',help="Need to self-define placeholder in the question")
    parser.add_argument('--filter', nargs='+',default=["1","2","3","4","5","6","7","32","33","34","35","36","38","39","40","41"], help="low vision filter id")
    # choices=[
    #     "liuhaotian/llava-v1.6-34b", "liuhaotian/llava-v1.5-13b",
    #     "liuhaotian/llava-v1.5-7b", "liuhaotian/llava-v1.6-mistral-7b",
    #     "liuhaotian/llava-v1.6-vicuna-13b", "liuhaotian/llava-v1.6-vicuna-7b"
    #                         ]
    
    args = parser.parse_args()
    model_name = 'llava'
    concept_file = args.infile.split('/')[-1]
    output_file = os.path.join(args.outfile,f'{concept_file}')

    model_name, tokenizer, model, image_processor, context_len = load_model(args.model_path, args.model_base)
    model_path = args.model_path
    
    # format:
    # a list of dict
    # minimum keys: image, question
    # [
    #     {
    #         "image": "xxxx",
    #         "question": "xxxxx"},
    #     ...]
    # leave the output key empty
    
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

            output = eval_model(model_name, tokenizer, model, image_processor, context_len, type('Args', (), {
                                    "model_path": model_path,
                                    "model_base": None,
                                    "model_name": model_name,
                                    "query": q,
                                    "conv_mode": None,
                                    "image_file": image_file,
                                    "sep": ",",
                                    "load_in_8bit": False,
                                    "load_in_4bit": False,
                                    "temperature": 0.0,  # set as 0.0 for reproceduce
                                    "top_p": None,
                                    "num_beams": 1,
                                    "max_new_tokens": 512
                                })())
            output = output.strip().replace(".", '')
            formatted_sample["rec_texts"] = output
            formatted_samples.append(formatted_sample)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True) 
    json.dump(formatted_samples, open(args.outfile, "w"), indent=4) 
            

