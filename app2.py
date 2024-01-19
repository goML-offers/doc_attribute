import streamlit as st
import openai
import base64
import PyPDF2
from io import BytesIO
from reportlab.pdfgen import canvas
import io
from fpdf import FPDF
from dotenv import load_dotenv
import os
import pinecone
import time
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import tempfile
import re
load_dotenv()

PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME') 
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')





def process_pdf(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file.read())
            temp_pdf_path=temp_pdf.name
        with open(temp_pdf_path, 'rb') as pdf_file:
            vectorize_and_store_documents(pdf_file.name)
    except Exception as e:
        st.error(f"Error processing PDF File: {e}")

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
    question=f"Pick up the major topics"
    search = docsearch.similarity_search(attribute)
    response = chain.run(input_documents=search, question=question)
    print("============")
    print(response)
    print("=============")

def call_attributes_api(formatted_prompt):
    try:
        answers = define_prompt(formatted_prompt)
        return answers

    except requests.exceptions.RequestException as attribute_error:
        st.error(f"Error in attributes request: {attribute_error}")
        return False
    

def generate_topics():
    st.info("Generating topics based on the PDF. Please wait...")

    topics = generate_topics_with_vectordb()

    return topics



def generate_topics_with_vectordb():

    # topics_list = pdf_text.split('\n')
    # formatted_topics = "\n".join([f"{i + 1}. {topic.strip()}" for i, topic in enumerate(topics_list)])

    # # Use OpenAI GPT-3 to generate topics
    # prompt = f"Generate topics based on the following PDF content:\n\n{formatted_topics}\n\nSelect topics by entering the corresponding numbers:"
    # response = openai.Completion.create(
    #     model="gpt-3.5-turbo-instruct",
    #     prompt=prompt[:4096],
    #     max_tokens=150,
    #     n=1,
    #     stop=None,
    #     temperature=0.5
    # )

    # # Extract selected topics from user input
    # selected_topics = response['choices'][0]['text'].strip().split('\n')
    
    # return selected_topics

    llm_chat = ChatOpenAI(temperature=0.5, max_tokens=50, model='gpt-3.5-turbo-0613', client='')
    embeddings = OpenAIEmbeddings(client='')
    docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    chain = load_qa_chain(llm_chat)
    answers = []
    question=f"List out 10-15 topics from the Table of contents enabling the user to query one of them , nature of the topics should be exhaustive  in nature with respect to the database with very little to nil common points to give diverse choice of topics for users to choose from also the topics should not have extra ( more than one) whitespaces and also no '\n and any other escape sequence character too.Igonre the underscores if any . Frame topics in a proper coherent and meaningful sentence.Pick up only the major topics from the pdf.Just separate each and every topic with '\n and remove the speacial  character and numbers like .,_- 0123456789 if any. "
    result = docsearch.similarity_search(question)
    # search = docsearch.similarity_search()
    # response = chain.run(input_documents=search,question=question)
    print("============")
    print(result)
    print("=============")
    # answers=list(response.split(',')
    # return answers
    answers=str(result[0])

    answers=answers.replace('_','').replace('.','').strip().split('\\n')
    pattern = re.compile(r'\b\d+\b')
    
    # Remove numbers from each string in the context
    context_without_numbers = [pattern.sub('', line) for line in answers]
    print(context_without_numbers)
    print(answers)

    return answers

def generate_pdf_with_selected_topics(selected_topics):
    # # For simplicity, let's create a PDF with the selected topics as content
    # prompt = f"Generate text on this topic:\n\n{selected_topics}\n\nTopics:"

    # response = openai.Completion.create(
    #     model="gpt-3.5-turbo-instruct",  # Use the latest supported engine
    #     prompt=prompt[:4096],  # Truncate prompt to fit within the model's maximum context length
    #     max_tokens=150,
    #     n=1,  # Number of topics to generate (adjust as needed)
    #     stop=None,
    #     temperature=0.5
    # )
    # print(response["choices"][0]['text'])
    # text = response["choices"][0]['text']
    # # pdf_content = "\n\n".join([f"Topic: {topic}" for topic in selected_topics])
    
    # pdf = FPDF()
    # pdf.add_page()
    # pdf.set_font("Arial", size=12)
    # pdf.multi_cell(0, 10, text)
    # return pdf.output(dest='S').encode('latin1')
    llm_chat = ChatOpenAI(temperature=0.5, max_tokens=2000, model='gpt-3.5-turbo-0613', client='')
    embeddings = OpenAIEmbeddings(client='')
    docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    chain = load_qa_chain(llm_chat)
    answers = []
    question=f" Retrieve all details related to {selected_topics} by performing a semantic search on the vector database , the nature of the result should be comprehensive in nature and should not out of the scope of the document , also  return the points in the structured format with meaningful interpretation. Explain me the {selected_topics}" 
    search = docsearch.similarity_search(selected_topics)
    response = chain.run(input_documents=search, question=question)
    print("============")
    print(response)
    print("=============")

    return response 



# def generate_pdf_with_selected_topics(selected_topics):
#     if not selected_topics:
#         return None

#     # For simplicity, let's create a PDF with the selected topics as content
#     pdf_content = "\n\n".join([f"Topic: {topic}" for topic in selected_topics])
#     return pdf_content


def generate_pdf(pdf_content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(100, 10, txt=pdf_content)
    return pdf.output("example.pdf")
    

def main():
    st.markdown(
        """
        <div style="display: flex; align-items: center; border-bottom: 2px solid green; padding-bottom: 1px; margin-bottom: 30px;">
            <h1 style="color: teal; margin: 0; font-family: 'Arial', sans-serif; font-size: 2.8em; font-weight: bold;">PDF Topic Generator</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        process_pdf(uploaded_file)
        topics = generate_topics()

        # Select topics and generate a new PDF
        selected_topics = st.selectbox("Select topics to include in the new PDF", topics)

        # if st.button("Generate PDF 📄"):
        pdf_content = generate_pdf_with_selected_topics(selected_topics)
        print("Generated pdf content is:")
        print(pdf_content)
        generate_pdf(pdf_content)
                

            # Download the generated PDF
        st.markdown(
                """
                <div>
                    <h1 style="color: teal; margin: 0; font-family: 'Arial', sans-serif; font-size: 1.8em; font-weight: bold;">Download Generated PDF</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
        with open("example.pdf", "rb") as f:
            st.download_button("Download pdf", f, "example.pdf")
if __name__ == "__main__":
    main()