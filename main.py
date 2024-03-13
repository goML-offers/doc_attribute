import streamlit as st
from openai import OpenAI
import base64
from PIL import Image, ImageDraw, ImageColor
import io
import ast
import numpy as np
api_key = "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"

client = OpenAI(api_key=api_key)


#function to get the damages value form the gpt output
def get_damages(content:str):

  descriptions_and_arrays = content.split("\n\n")

  damage_cv_arrays = []
  for item in descriptions_and_arrays:
      if "damage cv array:" in item:
          array_string = item.split("damage cv array:")[1].strip()
          damage_cv_arrays.append(ast.literal_eval(array_string))
  return damage_cv_arrays

def get_description(content:str):
  descriptions_and_arrays = content.split("\n\n")
  descriptions = []
  for item in descriptions_and_arrays:
      if "description:" in item:
            description = item.split("description:")[1].strip()
            # Remove "damage cv array:" and the subsequent sentence if present in the description
            description_parts = description.split("damage cv array:", 1)
            description = description_parts[0].strip()
            descriptions.append(description)

  print(descriptions)
  return descriptions

#function to encode the image

def encode_image(image):
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="JPEG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    return img_str
  
def highlight_defect(image, defect_array):
    draw = ImageDraw.Draw(image)
    red_color = (255, 0, 0, 50) 
    draw.rectangle(defect_array, outline=red_color, width=7)
    return image


   
# &&&&&&&&&&&&&&&&&&&&&&&&&Streamlit UI code&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
st.title("Inride Usecase")

# Upload file button
uploaded_files = st.file_uploader("Upload multiple JPEG or JPG files", accept_multiple_files=True)
file_names = []  # List to store the filenames

base64_images = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        base64_images.append(encode_image(image))
        file_names.append(uploaded_file.name)  # Store the filename
    print("meowwwwwww")
    print(file_names)
    print("meowwwwwww")
    # file_names.reverse()

if st.button("Submit"):
    if uploaded_files:
        for  i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            base64_image = encode_image(image)
            prompt = f"""you are give with {len(uploaded_files)} images of the cars image.By considering Base price: $20000
Scratch: Reduce valuation by 1%
Paint issues: Reduced by 1%
Tires damage: 0.5%
Tire grip damage: 0.5%
Dent: 1%
You need to tell each car exact base price after certain percentage of reduction based on damage by assuming base price as $20000 , create a array and store the the damages i need to hightlight the the particular area in the image 
i am using this open cv to highlight
draw.rectangle([(width//4, height//4), (width*3//4, height*3//4)], outline="red", width=2)

so give me the array of the damaged area so i can subittute over there in the below format (use exact values,no variable names with this format (width//4, height//4), (width*3//4, height*3//4))
in the below format 
output format:
image {i + 1}:  
description: (Identify the plate number of the car if any , Describe the damage here and the base price value after reduction)(Description has to be minimum 35 words,use same font style and size)
damage cv array:
(only the array value without any format, and no explaination)


Note:
if no damage found return a empty array, the array should exactly match the image when i subtitue the array in my function

Reference format: 
image 1:
description: (Describe the damage here)
damage cv array:
[(0, 0), (736, 150)]

image 2:
description: (Describe the damage here)
damage cv array:
[]

image 3:
description: (Describe the damage here)
damage cv array:
[(0, 0), (474, 238)]
"""
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                             {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            completion = response.choices[0].message.content
           
            print("Completion:", completion)
            damage_list = get_damages(completion)
            descriptions_list = get_description(completion)
            print("Description_lis",descriptions_list)
            print("damage_list",damage_list)
            if len(damage_list) > 0:
                for i, damage_array in enumerate(damage_list):
                    if len(damage_array) > 0:
                        image = highlight_defect(image, damage_list[i])
                        st.image(image)
                        st.write(f"Description: {descriptions_list[i]}")

                    else:
                        st.image(image)
                        st.write(f"Description: {descriptions_list[i]}")
                        file_names.pop(0)
            # else:
            #     st.write("No damage found in the uploaded images.")
