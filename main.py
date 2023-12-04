from fastapi import FastAPI, File, UploadFile
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


def vectorize_and_store_documents(pdf_filename):


    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    print('Checking if index exists...')
    
    


    if PINECONE_INDEX_NAME not in pinecone.list_indexes():

        
        print('Index does not exist, creating index...')
        # we create a new index
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            metric='cosine',
            # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
            dimension=1536
        )

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
    questions = [
        "Provide me the Insured Name. Do not add extra wordings. The name is given in the document. Check it iteratively and respond precisely. For example:Gokula Varshini Manikandan",
        "Provide me the Policy Number given in the document. Do not add extra wordings.Check it iteratively and respond precisely. For example:TIT/92033282",
        "Provide me the policy period . Do not add extra wordings.Check it iteratively and respond precisely. For Example: 29-APR-2022 to Midnight of 28-APR-2023",
        "Which company has provided the policy? Extract me only the company name: <<company name>>. Do not add extra wordings. For example: Third Party Plan"]
    # questions=["Provide me the Policy Number"]
    
    llm_chat = ChatOpenAI(temperature=0.5, max_tokens=50,
                         model='gpt-3.5-turbo-0613', client='')

    embeddings = OpenAIEmbeddings(client='')
    #Set Pinecone index
    docsearch = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    # Create chain
    chain = load_qa_chain(llm_chat)
    
    print("=========================")
    print(docsearch)
    print("==========================")
    answers = []
    for question in questions:
        search = docsearch.similarity_search(question)
        response = chain.run(input_documents=search, question=question)
        print(response)
        answers.append(response)
    print("empty++++++++++++++",docsearch)
    
    print(answers)
    
    return answers

@app.post("/upload-policy/")
async def upload_policy(pdf_file: UploadFile = File(...)):
    with open(pdf_file.filename, "wb") as f:
        f.write(pdf_file.file.read())
    
    answers=vectorize_and_store_documents(pdf_file.filename)
    os.remove(pdf_file.filename)
     
    # answers=extract_metadata()
    
    
    excel_filename = "answers_output.xlsx"
    if os.path.exists(excel_filename):
        existing_df = pd.read_excel(excel_filename)
    else:
        existing_df = pd.DataFrame(columns=["policy_holder_name", "policy_number", "policy_period", "policy_name"])
    new_data = pd.DataFrame([answers], columns=existing_df.columns)
    updated_df = pd.concat([existing_df, new_data], ignore_index=True)
    updated_df.to_excel(excel_filename, index=False)
    
    
    # index.delete(delete_all=True, namespace='gcp-starter')
    if PINECONE_INDEX_NAME in pinecone.list_indexes():
        print("deleting the existing index")
        pinecone.delete_index(PINECONE_INDEX_NAME)
        print("deleted")

    return {"message": "Answers extracted, Pinecone index deleted, and appended to Excel.", "file_name": excel_filename}


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
