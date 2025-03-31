import torch
import os
import argparse
import json
import tqdm
from PIL import Image
from collections.abc import Sequence
import re
from dotenv import load_dotenv
load_dotenv(".env")
torch.manual_seed(1234)


from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info






def eval_model(processor, model, messages):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output[0]




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str,default='data/viocr/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/qwen.json')
    parser.add_argument('--img_dir', type=str,default='data/viocr/selected_images_new',help="Directory storing images")
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument('--filter', nargs='+',default=["1","2","3","4","5","6","7","32","33","34","35","36","38","39","40","41"], help="low vision filter id")
    
    args = parser.parse_args()
    model_name = 'qwen2.5'
    concept_file = args.infile.split('/')[-1]
    output_file = os.path.join(args.outfile,f'{concept_file}')
    
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-3B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained(args.model_path)

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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_file,
                        },
                        {"type": "text",
                         "text": q},
                    ],
                }
            ]
            output = eval_model(processor, model, messages)
            output = output.strip()
            formatted_sample['rec_texts'] = output
            output_samples.append(formatted_sample)
    
        
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    json.dump(output_samples, open(args.outfile, "w"), indent=4)
            



