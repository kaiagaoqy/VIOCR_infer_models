export HF_HOME=/cis/net/io62a/data/qgao/hfcache
## --- MNREAD ---
# python model_zoo/infer_git_l.py --filter 0 --infile data/mnread/anno.json --outfile output/mnread/git_l.json --img_dir data/mnread
# python model_zoo/infer_git.py
# python model_zoo/infer_blip2.py
python model_zoo/infer_cogvlm.py --quant 4 --filter 0 --infile data/mnread/anno.json --outfile output/mnread/cogvlm.json --img_dir data/mnread
# python model_zoo/infer_claude.py --filter 0 --infile data/mnread/anno.json --outfile output/mnread/claude3_7_sonnet.json --model_path claude-3-7-sonnet-latest --img_dir data/mnread
# python model_zoo/infer_claude.py --filter 0 --infile data/mnread/anno.json --outfile output/mnread/claude3_5_haiku.json --model_path claude-3-5-haiku-latest --img_dir data/mnread
# python model_zoo/infer_gemini.py --filter 0 --infile data/mnread/anno.json --outfile output/mnread/gemini_15_flash.json --model_path gemini-1.5-flash --img_dir data/mnread
# python model_zoo/infer_gemini.py --filter 0 --infile data/mnread/anno.json --outfile output/mnread/gemini_2_flash.json --model_path gemini-2.0-flash --img_dir data/mnread
# python model_zoo/infer_gemini.py --filter 0 --infile data/mnread/anno.json --outfile output/mnread/gemini_15_pro.json --model_path gemini-1.5-pro --img_dir data/mnread
# python model_zoo/infer_llava.py --filter 0 --infile data/mnread/anno.json --outfile output/mnread/mplug.json --img_dir data/mnread
# python model_zoo/infer_mplug.py --filter 0 --infile data/mnread/anno.json --outfile output/mnread/mplug.json --img_dir data/mnread
# python model_zoo/infer_qwen.py  --filter 0 --infile data/mnread/anno.json --outfile output/mnread/qwen.json --img_dir data/mnread
# python model_zoo/infer_gpt.py --filter 0 --infile data/mnread/anno.json --outfile output/mnread/gpt.json --img_dir data/mnread --model_path gpt4o
# python model_zoo/infer_gpt.py --filter 0 --infile data/mnread/anno.json --outfile output/mnread/gpt4o_mini.json --img_dir data/mnread --model_path gpt-4o-mini

## --- ETDRS ---
# python model_zoo/infer_git_l.py --filter 0 --infile data/etdrs/anno.json --outfile output/etdrs/git_l.json --img_dir data/etdrs
# python model_zoo/infer_git.py --filter 0 --infile data/etdrs/anno.json --outfile output/etdrs/git.json --img_dir data/etdrs
# python model_zoo/infer_blip2.py --filter 0 --infile data/etdrs/anno.json --outfile output/etdrs/blip2.json --img_dir data/etdrs
python model_zoo/infer_cogvlm.py --quant 4 --filter 0 --infile data/etdrs/anno.json --outfile output/etdrs/cogvlm.json --img_dir data/etdrs
# python model_zoo/infer_claude.py --filter 0 --infile data/etdrs/anno.json --outfile output/etdrs/claude3_7_sonnet.json --model_path claude-3-7-sonnet-latest --img_dir data/etdrs
# python model_zoo/infer_claude.py --filter 0 --infile data/etdrs/anno.json --outfile output/etdrs/claude3_5_haiku.json --model_path claude-3-5-haiku-latest --img_dir data/etdrs

# python model_zoo/infer_gemini.py --filter 0 --infile data/etdrs/anno.json --outfile output/etdrs/gemini_15_flash.json --model_path gemini-1.5-flash --img_dir data/etdrs
# python model_zoo/infer_gemini.py --filter 0 --infile data/etdrs/anno.json --outfile output/etdrs/gemini_2_flash.json --model_path gemini-2.0-flash --img_dir data/etdrs
# python model_zoo/infer_gemini.py --filter 0 --infile data/etdrs/anno.json --outfile output/etdrs/gemini_15_pro.json --model_path gemini-1.5-pro --img_dir data/etdrs
# python model_zoo/infer_llava.py
python model_zoo/infer_mplug.py --filter 0 --infile data/etdrs/anno.json --outfile output/etdrs/mplug.json --img_dir data/etdrs
# python model_zoo/infer_qwen.py  --filter 0 --infile data/etdrs/anno.json --outfile output/etdrs/qwen.json --img_dir data/etdrs
# python model_zoo/infer_gpt.py --filter 0 --infile data/etdrs/anno.json --outfile output/etdrs/gpt.json --img_dir data/etdrs
# python model_zoo/infer_gpt.py --filter 0 --infile data/etdrs/anno.json --outfile output/etdrs/gpt4o_mini.json --img_dir data/etdrs --model_path gpt-4o-mini