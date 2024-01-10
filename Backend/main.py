from fastapi import FastAPI, File, UploadFile
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
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


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

load_dotenv()

PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')  # 'kn1'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

class AttributeInput(BaseModel):
    formatted_prompt: str

def vectorize_and_store_documents(pdf_filename):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    print('Checking if index exists...')
    

    if PINECONE_INDEX_NAME in pinecone.list_indexes():
        print("deleting the existing index")
        pinecone.delete_index(PINECONE_INDEX_NAME)
        print("deleted")

    time.sleep(20)

    if PINECONE_INDEX_NAME not in pinecone.list_indexes(): 
        print('Index does not exist, creating index...')
        # we create a new index
        pinecone.create_index(name=PINECONE_INDEX_NAME,metric='cosine',dimension=1536)
        #pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    # Load PDF documents
    loader = PyPDFLoader(pdf_filename)
    pages = loader.load()
    # Chunk data into smaller documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(pages)
    print(texts)

     # Vectorize documents using OpenAI embeddings
    embeddings = OpenAIEmbeddings(client='')
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=PINECONE_INDEX_NAME)
    print(docsearch)
    time.sleep(20)



def define_prompt(formatted_prompt):
    llm_chat = ChatOpenAI(temperature=0.5, max_tokens=50, model='gpt-3.5-turbo-0613', client='')
    embeddings = OpenAIEmbeddings(client='')
    docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    chain = load_qa_chain(llm_chat)
    answers = []
    attributes=[attr.strip() for attr in formatted_prompt.split(",")]


    for attribute in attributes:
        question=f"Provide me the {attribute}. Do not add extra wordings.Check it iteratively and respond precisely. Add the confidence scale from scale 0 to 1.Provide me in two line For example: policy period :20 Apr 23 00:00 hrs to 19 Apr 25 23:59 hrs. \n Confidence level:0.9 . colon to show the value is important "
        search = docsearch.similarity_search(attribute)
        response = chain.run(input_documents=search, question=question)
        print("============")
        print(response)
        print("=============")
        
        lines = response.strip().split('\n')
        if len(lines) >= 2:
            attribute_value = lines[0].split(':',1)[1].strip()
            confidence_value = lines[1].split(':')[1].strip()
            answers.append({"attribute": attribute,"value":attribute_value, "confidence": float(confidence_value)})
        else:
            answers.append({attribute: "N/A", "confidence": 0.0})  # Handle cases where lines are not present
        # answers.append({"attribute":attribute})
    
    print(answers)
    return answers

@app.post("/upload-policy/")
async def upload_policy(pdf_file: UploadFile = File(...)):
    with open(pdf_file.filename, "wb") as f:
        f.write(pdf_file.file.read())
    
    vectorize_and_store_documents(pdf_file.filename)
    os.remove(pdf_file.filename)

@app.post("/attributes/")
async def attributes(input_data: AttributeInput):
    formatted_prompt = input_data.formatted_prompt
    answers = define_prompt(formatted_prompt)
    return answers

    # return {"message": "Answers extracted, Pinecone index deleted, and appended to Excel.", "file_name": excel_filename}
# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
