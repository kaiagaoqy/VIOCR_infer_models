export HF_HOME=/cis/net/io62a/data/qgao/hfcache
#python model_zoo/infer_git_l.py
#python model_zoo/infer_git.py
#python model_zoo/infer_blip2.py
python model_zoo/infer_cogvlm.py --quant 4
# python model_zoo/infer_claude.py
python model_zoo/infer_gemini.py
python model_zoo/infer_llava.py
# python model_zoo/infer_mplug.py
python model_zoo/infer_qwen.py
python model_zoo/infer_gpt.py