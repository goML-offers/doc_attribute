import streamlit as st
import pandas as pd
import requests
import tempfile
import time
import json
import base64
from typing import List
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os
import pandas as pd
from dotenv import load_dotenv
import pinecone
import time
import pdfplumber
import io
import xlsxwriter
from io import BytesIO 
import fitz 
from PIL import Image
import shutil
import easyocr
import cv2
import numpy as np
import pytesseract
import argparse
import keras_ocr
import matplotlib.pyplot as plt


buffer = io.BytesIO()
buffer = io.BytesIO()

load_dotenv()

PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME') 
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')



if "edited_df" not in st.session_state:
    st.session_state.edited_df = pd.DataFrame()

if 'ctr' not in st.session_state:
    st.session_state['ctr'] = 0

if "result" not in st.session_state:
    st.session_state.result = ''

if 'file_uploader_pdf' not in st.session_state:
    st.session_state.file_uploader_pdf = None

if 'file_uploader_excel' not in st.session_state:
    st.session_state.file_uploader_excel = None


def vectorize_and_store_documents(pdf_filename,vision_out,test_extract_output):
    # pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    # print('Checking if index exists...')
    

    # if PINECONE_INDEX_NAME in pinecone.list_indexes():
    #     print("deleting the existing index")
    #     pinecone.delete_index(PINECONE_INDEX_NAME)
    #     print("deleted")

    # time.sleep(20)

    # if PINECONE_INDEX_NAME not in pinecone.list_indexes(): 
    #     print('Index does not exist, creating index...')
    #     # we create a new index
    #     pinecone.create_index(name=PINECONE_INDEX_NAME,metric='cosine',dimension=1536)
    #     #pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    # Load PDF documents
    loader = PyPDFLoader(pdf_filename)
    pages = loader.load()
    # Chunk data into smaller documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    texts_from_pdf = text_splitter.split_documents(pages)

    texts = [t.page_content for t in texts_from_pdf] + vision_out+ list(test_extract_output)

    # texts2=text_splitter.split_documents(vision_out)
    # texts+=texts2
    # print(texts)
    
    return texts
    embeddings = OpenAIEmbeddings(client='')
    docsearch = Pinecone.from_texts(texts, embeddings, index_name=PINECONE_INDEX_NAME)
    print(docsearch)
    time.sleep(20)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_text_easy_ocr(data):
    extracted_text = []
    for entry in data:
        text = entry[1]  # Extracting the text from the tuple
        extracted_text.append(text)
    return extracted_text

def tessract(pdf_file_name):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # img = cv2.imread(f'output_images/{pdf_file_name}1.png')
    image = cv2.imread(f'output_images/{pdf_file_name}1.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(image)
    print(text)
    return text 

# def keras_ocr_extract(pdf_file_name):
#     pipeline = keras_ocr.pipeline.Pipeline()
#     image_path = f'output_images/{pdf_file_name}1.png'
#     image = keras_ocr.tools.read(image_path)
#     prediction_group = pipeline.recognize([image])[0]

#     fig, ax = plt.subplots(figsize=(10, 20))
#     keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_group, ax=ax)
#     text_list = []
#     for text, box in prediction_group:
#         print("........keras.........", text)
#         text_list.append(text)
#     print(".................final",text_list)
#     return text_list
def vision_bot(pdf_file_name):
    results=[]
    print("1v")
    image_paths=[f'output_images/{pdf_file_name}1.png']
    # Iterate through each image path
    print("2v")
    for image_path in image_paths:
        # Getting the base64 string

        print(image_path)
        print("3v")
        base64_image = encode_image(image_path)
        print("4v")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """perform a deep OCR try to interpret all feilds and values you see in this form.  you only job is to provide absolutely accurate 
                            information, so do a detail and accurate ocr, and all the fields I have given are present in the document image, if you're not able to find, 
                            again do an ocr, but you must give accurate and all fields. you are provided with full rights to extract the details in the document.
                            If you are not able to run a dep ocr just bring out only few details from the document.Just provide the output file as json file for example 
                            {  "Customer Number": "8417",  "Customer PO Number": "0051566",  "Salesperson": "Eric Thompson",  "Ship Via": "OT OUR TRUCK",  "Ship Date": "09/06/2023",  "Order Qty": "1ea",  "Ship Qty": "1ea",  "Description": "S-A B330H 30A 3P 240V 22KA",  "Payments Amount Due": "NA"}"""
                            
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
        "max_tokens":2000
        }
        print("5v")
        # Make the API request
        temp =[]
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print("6v")
        temp.append(response.json())
        print("7v")
        result = temp[0]
        print("8v")
        message = result["choices"][0]["message"]["content"]
        print("9v")
        results.append(message)
        print("10v")
        return results


def vanilla_openai(results_json,formatted_prompt,output,tessract):
    results=[]
    query=f"Your task is to act as an expert data extractor You will be given documents containing various data points. Extract the following fields and present them in a JSON format:{formatted_prompt} For each field, ensure accuracy and completeness. If a field is not present or its value is unclear, indicate this with 'NA'.Conduct thorough checks to ensure the accuracy of the data extracted. Your main objective is to provide clear, accurate, and well-structured data in JSON format.Dont return any wordings before or after the json output. Your job is to return only the json file"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":f"{tessract},{output},{results_json},{query}",
                    },
                    
                ]
            }
        ],
    "max_tokens":2000
    }

    # Make the API request
    temp =[]
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    temp.append(response.json())
    result = temp[0]
    message = result["choices"][0]["message"]["content"]
    results.append(message)
    print("===========================================================================")
    print(results)
    print("===========================================================================")
    return results



def define_prompt(formatted_prompt):
    llm_chat = ChatOpenAI(temperature=0.6, max_tokens=50, model='gpt-4', client='')
    embeddings = OpenAIEmbeddings(client='')
    docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    chain = load_qa_chain(llm_chat)
    answers = []
    attributes=[attr.strip() for attr in formatted_prompt.split(",")]


    for attribute in attributes:
        question=f"""I will be giving you set of documents that have complex data and I want you to act as a data extractor.Provide me the {attribute} value from the document.
        Check it iteratively and respond precisely. Add the confidence scale from scale 0 to 1.
        Provide me in two line For example: policy period :20 Apr 23 00:00 hrs to 19 Apr 25 23:59 hrs. \n Confidence level:0.9 . colon to show the value is important.
        If you don't know the answer or answer doesnot exist the just give attribute as attribute name and value as NA. """
        search = docsearch.similarity_search(attribute)
        response = chain.run(input_documents=search, question=question)
        print("============")
        print(response)
        print("=============")
        
        lines = response.strip().split('\n')
        if len(lines) >= 2:
            attribute_value = lines[0].split(':',1)[1].strip()
            confidence_value = lines[1].split(':')[1].strip()
            answers.append({"attribute": attribute,"value":attribute_value})
        else:
            answers.append({"attribute": attribute,"value":"N/A"})  # Handle cases where lines are not present
        # answers.append({"attribute":attribute})
    
    print(answers)
    return answers

def upload_policy(pdf_file):
    try:
        with open(pdf_file, "wb") as f:
            f.write(pdf_file.file.read())
    
        vectorize_and_store_documents(pdf_file)
        os.remove(pdf_file)


    except requests.exceptions.RequestException as policy_error:
        st.error(f"Error in policy upload request: {policy_error}")
        return False

def call_attributes_api(formatted_prompt):
    try:
        answers = define_prompt(formatted_prompt)
        return answers

    except requests.exceptions.RequestException as attribute_error:
        st.error(f"Error in attributes request: {attribute_error}")
        return False

# def save_to_json(dataframe, filename="output.json"):
#     with open(filename, 'w') as json_file:
#         json.dump(dataframe.to_json(orient='records'), json_file)

# def on_download_click(dataframe, filename):
#     save_to_json(dataframe, filename)
#     with open(filename, 'r') as file:
#         data = file.read()
#     b64 = base64.b64encode(data.encode()).decode()
#     href = f'<a href="data:file/json;base64,{b64}" download="{filename}">Click here to download the file</a>'
#     st.markdown(href, unsafe_allow_html=True)
#     time.sleep(5)
#     st.success("File downloaded successfully!")

def uploader_callback():
    # Update session state keys
    st.session_state.file_uploader_pdf = uploaded_file_pdf
    st.session_state.file_uploader_excel = uploaded_file_excel

    if st.session_state.file_uploader_pdf is not None:
        browse_pdf_files(st.session_state.file_uploader_pdf)

    if st.session_state.file_uploader_excel is not None:
        browse_excel_file(st.session_state.file_uploader_excel)

def extract_text_from_pdf(pdf_path):
    print("PDF Path:", pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    print("extract text from pdf ",text) 
    return text

# def browse_pdf_files(pdf_files, excel_file):
#     result_dfs = []
#     i=0

#     for pdf_file in pdf_files:
#         print(pdf_file)
#         print(len(pdf_files))
#         try:
#             # Save the PDF buffer to a temporary file
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#                 temp_pdf.write(pdf_file.read())
#                 temp_pdf_path = temp_pdf.name
            
#             # Save the Excel buffer to a temporary file
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_excel:
#                 temp_excel.write(excel_file.read())
#                 temp_excel_path = temp_excel.name

#             with open(temp_pdf_path, 'rb') as pdf_file:
#                 vectorize_and_store_documents(pdf_file.name)
#             uploaded_df = pd.read_excel(temp_excel_path, engine='openpyxl')

#             attributes_str = ",".join(map(str, uploaded_df.columns))
#             print(attributes_str)

#             formatted_prompt_input = attributes_str.strip()

#             result = call_attributes_api(formatted_prompt_input)

#             for item in result:
#                 attribute_name = item['attribute']
#                 attribute_value = item['value']
#                 uploaded_df.loc[i, attribute_name] = f"{attribute_value}"

#             result_dfs.append(uploaded_df)
            
#             i+=1
#         except Exception as e:
#             st.error(f"Error processing PDF File '{pdf_file.name}': {e}")

#     if result_dfs:
#         final_result_df = pd.concat(result_dfs, ignore_index=True)
#         return final_result_df
#     else:
#         return None




def process_pdf(vision_out,pdf_file,excel_df,i):
    
        # Save the PDF buffer to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file.read())
            temp_pdf_path = temp_pdf.name
        test_extract_output = extract_text_from_pdf(temp_pdf_path)
        # Process the PDF and update the Excel DataFrame
        with open(temp_pdf_path, 'rb') as pdf_file:
            results_json=vectorize_and_store_documents(pdf_file.name,vision_out,test_extract_output)
        
        

        attributes_str = ",".join(map(str, excel_df.columns))
        formatted_prompt_input = attributes_str.strip()
        print("formateddddddddd",formatted_prompt_input)
        return results_json,list(attributes_str)
        # result = call_attributes_api(formatted_prompt_input)
        # print("resultsssss",result)
        # for item in result:
        #     attribute_name = item['attribute']
        #     attribute_value = item['value']
        #     excel_df.loc[i, attribute_name] = f"{attribute_value}"
        # print(excel_df)
       
        # return excel_df

    # except Exception as e:
    #     st.error(f"Error processing PDF File: {e}")
    #     return excel_df


def browse_pdf_files(vision_out,pdf_files, excel_file,output,tessract):
    result_df = pd.read_excel(excel_file, engine='openpyxl') if excel_file else pd.DataFrame()
    i=0
    for pdf_file in pdf_files:
        result_json,formatted_prompt = process_pdf(vision_out,pdf_file, result_df,i)
        i+=1
    final_result=vanilla_openai(result_json,formatted_prompt,output,tessract)
    return final_result


def pdf_to_images(pdf_file, output_folder,pdf_file_name):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the PDF file from BytesIO using fitz.open
    pdf_document = fitz.open(stream=pdf_file)

    # Iterate through each page in the PDF
    for page_number in range(pdf_document.page_count):
        # Get the page
        page = pdf_document[page_number]

        # Get the Pixmap for the page
        pixmap = page.get_pixmap()

        # Convert the Pixmap to a bytes object
        img_bytes = pixmap.samples

        # Use PIL to open the image from bytes
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], img_bytes)

        # Save the image to the output folder
        image_path = os.path.join(output_folder, f"{pdf_file_name}{page_number + 1}.png")
        image.save(image_path, "PNG")

    # Close the PDF file
    pdf_document.close()

# def browse_pdf_files(vision_out,pdf_files, excel_file):
#     result_df = pd.read_excel(excel_file, engine='openpyxl') if excel_file else pd.DataFrame()
#     i=0

#     for pdf_file in pdf_files:
        
#         sheet_name = os.path.splitext(os.path.basename(pdf_file.name))[0]
#         print(sheet_name)

#         # result_df=process_pdf(pdf_file, sheet_name, i,result_df)
#         result_df = process_pdf(vision_out,pdf_file, result_df,i)
#         i+=1

    # with pd.ExcelWriter(excel_file, engine='openpyxl') as excel_writer:
    #     for pdf_file in pdf_files:
    #         sheet_name = os.path.splitext(os.path.basename(pdf_file.name))[0]
    #         result_df = pd.read_excel(excel_file, sheet_name, engine='openpyxl') if excel_file else pd.DataFrame()
    #         print(sheet_name)

    #         result_df = process_pdf(pdf_file, sheet_name, i, result_df)
    #         result_df.to_excel(excel_writer, sheet_name, index=False)  # Write to the Excel file
    #         i += 1
 




logo_url = "data:image/jpeg;base64,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" 

col1, col2 = st.columns([1, 3])
with col1:
    st.image(logo_url, width=70)

with col2:
    st.title("MetaData Extraction")

uploaded_file_pdf = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_file_uploader", accept_multiple_files=True)
uploaded_file_excel = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"], key="excel_file_uploader")

st.subheader("MetaData")

uploaded_file_pdf_2 = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_file_uploader_2", accept_multiple_files=True)
with st.spinner("Processing..."):
    if st.button("Execute"):
        if uploaded_file_pdf and uploaded_file_excel:
            vision_out=""
            a=uploaded_file_pdf
            output_folder = "output_images"
            for b in a:
                pdf_file_name =b.name
                temp_pdf_file = BytesIO(b.read())
                temp_pdf_file.seek(0)
                if temp_pdf_file.readable() and temp_pdf_file.read(1):
                    temp_pdf_file.seek(0)  # Reset file pointer to the beginning
                    pdf_to_images(temp_pdf_file, output_folder, pdf_file_name)
                    vision_out = vision_bot(pdf_file_name)
                    print(vision_out)
                else:
                    print(f"Skipping empty PDF file: {pdf_file_name}")
                        # pdf_to_images(temp_pdf_file, output_folder,pdf_file_name)
            image_file=f"output_images/{pdf_file_name}1.png"
            reader = easyocr.Reader(['en'])
            output = reader.readtext(image_file)
            output = extract_text_easy_ocr(output)
            print("output",output)
            # vision_out=vision_bot(pdf_file_name)
            # print(vision_out)
            
            tess_output=tessract(pdf_file_name)
            print("tessssoutput",tess_output)

            # keras_output=keras_ocr_extract(pdf_file_name)

        try:
            print("1")

            result_json = browse_pdf_files(vision_out,uploaded_file_pdf_2, uploaded_file_excel,output,tess_output)
            print("2")
            # st.write(result_json)
            # match = re.search(r'{(.*?)}', result_json, re.DOTALL)

            # extracted_json_string = match.group(1)
            json_string=result_json[0]
            print("3")
            json_string = json_string.replace('```json\n', '').replace('```', '').replace('\n', '')
            print("4")
            json_string=json_string.strip()
            print(json_string)
            print("5")
            result_dict = json.loads(json_string)
            
            print("6")
            # Create a DataFrame from the flattened dictionary
            df = pd.DataFrame([result_dict])
            # df = pd.DataFrame([result_json])
            print("7")
            st.write(df)
            # Save DataFrame to Excel file
            csv_filename = 'output.csv'
            absolute_path = os.path.abspath(csv_filename)
            print("8")
            df.to_csv(absolute_path, index=False)
            # result_df = browse_pdf_files(vision_out,uploaded_file_pdf, uploaded_file_excel)
            # print(result_df)
            # if result_df is not None:
            #     st.data_editor(result_df)
            #     st.session_state.result = result_df

            #     result_df.to_excel(buffer, index=False, engine='openpyxl')

            #     buffer.seek(0)
            # os.rmdir("output_images")
            print("9")
            shutil.rmtree("output_images")
            print("10")
            st.download_button(
            label="Download CSV file",
            data=open(absolute_path, 'rb').read(),
            file_name="output.csv",
            mime="text/csv"
        )
            print("11")


        except Exception as e:
            print(f"Error processing PDF files: {e}")
            st.subheader("I faced an issue fetching the data. This could be due to an error in the uploaded file, requested fields of my ML model not responding. Please try again!")
       