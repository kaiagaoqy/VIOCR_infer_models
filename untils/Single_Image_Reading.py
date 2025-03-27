from pathlib import Path
import base64
import requests
import glob
# OpenAI API Key
api_key = "OpenAI key"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
file_path = "/Users/Ishaan Chaturvedi/Downloads/selected_images/1"
image_files = Path(file_path)
for image_path in image_files.glob("*.jpg"):
	# Path to your image
	print(image_path)

	# Getting the base64 string
	base64_image = encode_image(image_path)
	headers = {
	  "Content-Type": "application/json",
	  "Authorization": f"Bearer {api_key}"
	}

	payload = {
	  "model": "gpt-4o",
	  "messages": [
	    {
	      "role": "user",
	      "content": [
	        {
	          "type": "text",
	          "text": "Read all the visibile text in the image."
	        },
	        {
	          "type": "image_url",
	          "image_url": {
	          "url": f"data:image/jpg;base64,{base64_image}",
		  "detail": "auto"
	          },
	        },
	      ],
	    }
	  ],
	  "max_tokens": 300
	}

	response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

	print(response.json())
