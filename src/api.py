from openai import OpenAI


# Create a service instance where you can set the base_url, replace the key with the purchased high-speed key from Zhiyun
client = OpenAI( base_url= "#",
    api_key = "#")


import base64
import requests


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = '/root/gptForMed/standard.png'
query_path = '/root/gptForMed/data.jpg'

# Getting the base64 string
base64_image = encode_image(image_path)
query_base64_image = encode_image(query_path)
# base64_image

messages=[
  {
    "role": "user",
    "content": [
      {"type": "text",
       # "text": "Describe the content of the two images in Chinese, using the rules of the first image to describe the details of the second image"},
       "text": "These two images, one is the standard for judgment, and the other is the sample to be judged. Provide the judgment and the corresponding probability distribution. The output example is: normal, 0.4, 0.2, 0.3, 0.1"},
      {
        "type": "image_url",
        "image_url": {
           "url": f"data:image/jpeg;base64,{base64_image}"
        },
      },
      {
        "type": "image_url",
        "image_url": {
           "url": f"data:image/jpeg;base64,{query_base64_image}"
        },
      },
    ],
  }
]

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=messages,
  max_tokens=300,
)

print(response)