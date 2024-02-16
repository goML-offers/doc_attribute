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
from streamlit_option_menu import option_menu
import psycopg2
from streamlit_authenticator import Authenticate
import hashlib
import mysql.connector as sql


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
    st.session_state.result = None


if "user_id" not in st.session_state:
    st.session_state.user_id=0

if "workspace_name" not in st.session_state:
    st.session_state.workspace_name = ""

if "selected" not in st.session_state:
    st.session_state.selected = None  
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

if 'gpt_key' not in st.session_state:
    st.session_state.gpt_key=""




#Function to create a workspace
def create_workspace(id, workspace_name,user_id):
    connection_params = {
        'host': 'database-1.cmeaoe1g4zcd.ap-south-1.rds.amazonaws.com',
        'port': '5432',
        'database': 'postgres',
        'user': 'postgres',
        'password': 'postgres'
    }

    connection = psycopg2.connect(**connection_params)
    cursor = connection.cursor()
    workspace_created = False
    try:
        cursor.execute("INSERT INTO genpact.workspaces (id, workspace_name,user_id) VALUES (%s, %s,%s) RETURNING id;", (id, workspace_name,user_id))
        workspace_id = cursor.fetchone()[0]
        connection.commit()
        st.success(f'Workspace "{workspace_name}" created successfully .')
        workspace_created = True
    except psycopg2.errors.UniqueViolation as e:

        connection.rollback()
        st.info(f'Workspace "{workspace_name}" already exists. Please choose a different workspace name.')
        workspace_id = None

    except Exception as e:
        connection.rollback()
        st.warning(f'Error creating workspace: {e}')
        workspace_id = None
    finally:
        connection.commit()
        connection.close()

    return workspace_created,workspace_id,user_id

def create_dynamic_table(table_name, columns, data):
    connection_params = {
        'host': 'database-1.cmeaoe1g4zcd.ap-south-1.rds.amazonaws.com',
        'port': '5432',
        'database': 'postgres',
        'user': 'postgres',
        'password': 'postgres'
    }

    connection = psycopg2.connect(**connection_params)
    cursor = connection.cursor()
    columns_lower = [col.lower().replace(' ', '_') for col in columns]
    column_definitions = ", ".join([f'"{col}" VARCHAR' for col in columns_lower])
    
    create_table_query = f"CREATE TABLE IF NOT EXISTS genpact.{table_name} ({column_definitions});"
    # create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_definitions)});"
    cursor.execute(create_table_query)
    values = "', '".join(str(row[col]) for col in columns_lower if col in row.index)
    insert_query = f"INSERT INTO genpact.{table_name} ({', '.join(columns_lower)}) VALUES ('{values}');"
    print(insert_query)
    cursor.execute(insert_query)

    connection.commit()
    connection.close()


def inser_metadata(workspace_name,filename,user_id,formatted_prompt_list,metadata_name):
    connection_params = {
        'host': 'database-1.cmeaoe1g4zcd.ap-south-1.rds.amazonaws.com',
        'port': '5432',
        'database': 'postgres',
        'user': 'postgres',
        'password': 'postgres'
    }

    connection = psycopg2.connect(**connection_params)
    cursor = connection.cursor()

    print("User id inside get_workspace_history",user_id)
    # Retrieve workspace history
    metadata_convert=str(formatted_prompt_list)

    metadata = json.dumps(metadata_convert)
    print(metadata)
    filename=filename.replace(' ','_')

    query = "INSERT INTO genpact.workspace_history (workspace_name, filename, user_id, metadata,metadata_name) VALUES (%s, %s, %s, %s,%s);"
    data = (workspace_name, filename, user_id, metadata,metadata_name)
    
    print(query)
    cursor.execute(query, data)
    st.session_state.user_id = user_id

    connection.commit()
    connection.close()

def get_metadata_values(user_id):
    connection_params = {
        'host': 'database-1.cmeaoe1g4zcd.ap-south-1.rds.amazonaws.com',
        'port': '5432',
        'database': 'postgres',
        'user': 'postgres',
        'password': 'postgres'
    }

    connection = psycopg2.connect(**connection_params)
    cursor = connection.cursor()

    # Fetch distinct metadata values from workspace_history
    cursor.execute(f"SELECT DISTINCT metadata_name FROM genpact.workspace_history where user_id={user_id};")
    metadata_values = [row[0] for row in cursor.fetchall()]
    connection.commit()
    connection.close()
    metadata_values = ['None' if val == "[\'\']" else val for val in metadata_values]

    print("....",metadata_values)
    return metadata_values

def update_gptkey(user_id,gpt_key):
    connection_params = {
        'host': 'database-1.cmeaoe1g4zcd.ap-south-1.rds.amazonaws.com',
        'port': '5432',
        'database': 'postgres',
        'user': 'postgres',
        'password': 'postgres'
    }

    connection = psycopg2.connect(**connection_params)
    cursor = connection.cursor()

    # Fetch distinct metadata values from workspace_history
    cursor.execute(f"UPDATE genpact.workspaces SET gptkey = '{gpt_key}' WHERE user_id = {user_id};")
    
    connection.commit()
    connection.close()




# Function to retrieve workspace history
def get_workspace_history(user_id):
    connection_params = {
        'host': 'database-1.cmeaoe1g4zcd.ap-south-1.rds.amazonaws.com',
        'port': '5432',
        'database': 'postgres',
        'user': 'postgres',
        'password': 'postgres'
    }

    connection = psycopg2.connect(**connection_params)
    cursor = connection.cursor()

    print("User id inside get_workspace_history",user_id)
    # Retrieve workspace history
    cursor.execute(f"SELECT DISTINCT workspace_name FROM genpact.workspace_history WHERE user_id={user_id};")
    workspace_history = cursor.fetchall()
    st.session_state.user_id = user_id

    print("workspace_history ",workspace_history)
    connection.commit()
    connection.close()

    return workspace_history



def insert_user_credentials(username, password):
    connection_params = {
        'host': 'database-1.cmeaoe1g4zcd.ap-south-1.rds.amazonaws.com',
        'port': '5432',
        'database': 'postgres',
        'user': 'postgres',
        'password': 'postgres'
    }

    try:
        connection = psycopg2.connect(**connection_params)
        cursor = connection.cursor()


        hashed_password = hash_password(password)

        cursor.execute("INSERT INTO genpact.users (username, password) VALUES (%s, %s);", (username, hashed_password))

        connection.commit()
        cursor.close()
        connection.close()

        return True
    except Exception as e:
        print(f"Error inserting user credentials: {e}")
        return False
    
def hash_password(password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    print("*********",hashed_password)
    return hashed_password

def fetch_user_credentials(username, password):
    connection_params = {
        'host': 'database-1.cmeaoe1g4zcd.ap-south-1.rds.amazonaws.com',
        'port': '5432',
        'database': 'postgres',
        'user': 'postgres',
        'password': 'postgres'
    }

    try:
        connection = psycopg2.connect(**connection_params)
        cursor = connection.cursor()

        # Fetch hashed password from the 'users' table for the provided username
        cursor.execute("SELECT user_id,password FROM genpact.users WHERE username = %s;", (username,))
        user_data = cursor.fetchone()
        
        print("userdata",user_data)
        cursor.close()
        connection.close()
        user_id,hashed_password=user_data
        print("User id inside the fetch_user_credentials",user_id)
  
        print(verify_password(password, hashed_password))
        if verify_password(password, hashed_password):
            return True,user_id
        else:
            st.warning("Incorrect username or password")
            return False,user_id

    except Exception as e:
        print(f"Error fetching user credentials: {e}")
        return False,user_id
    
def verify_password(input_password, hashed_password):
    print(input_password,hashed_password,hash_password(input_password))
    print(type(hashed_password),type(hash_password(input_password)))
    return hash_password(input_password) == hashed_password



def vectorize_and_store_documents(pdf_filename):
    # OPENAI_API_KEY=f"{gpt_key}"
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



def define_prompt(formatted_prompt,model):
    # OPENAI_API_KEY=f"{gpt_key}"
    llm_chat = ChatOpenAI(temperature=0.5, max_tokens=50, model=model, client='')
    embeddings = OpenAIEmbeddings(client='')
    docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    chain = load_qa_chain(llm_chat)
    answers = []
    attributes=[attr.strip() for attr in formatted_prompt.split(",")]


    for attribute in attributes:
        question=f"Provide me the {attribute}. Do not add extra wordings.Check it iteratively and respond precisely. Add the confidence scale from scale 0 to 1.Add the descriptive answer to the attribute and value.Provide me in three line For example: policy period :20 Apr 23 00:00 hrs to 19 Apr 25 23:59 hrs. \n Confidence level:0.9 \n Description: The policy period of the insurance is from 20th april 2023 to 19th april 2025. colon to show the value is important "
        search = docsearch.similarity_search(attribute)
        response = chain.run(input_documents=search, question=question)
        print("============")
        print(response)
        print("=============")
        
        lines = response.strip().split('\n')
        if len(lines) >= 2:
            attribute_value = lines[0].split(':',1)[1].strip()
            confidence_value = lines[1].split(':')[1].strip()
            description = lines[2].split(':')[1].strip()
            answers.append({"attribute": attribute,"value":attribute_value, "confidence": float(confidence_value),"Description":description})
        else:
            answers.append({attribute: "N/A", "confidence": 0.0,'description':"N/A"})  # Handle cases where lines are not present
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

def call_attributes_api(formatted_prompt,selected_model):
    try:
        answers = define_prompt(formatted_prompt,selected_model)
        return answers

    except requests.exceptions.RequestException as attribute_error:
        st.error(f"Error in attributes request: {attribute_error}")
        return False

def save_to_json(dataframe, filename="output.json"):
    with open(filename, 'w') as json_file:
        json.dump(dataframe.to_json(orient='records'), json_file)

def on_download_click(dataframe, filename):
    save_to_json(dataframe, filename)
    with open(filename, 'r') as file:
        data = file.read()
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">Click here to download the file</a>'
    st.markdown(href, unsafe_allow_html=True)
    time.sleep(5)
    st.success("File downloaded successfully!")

def uploader_callback():
    if st.session_state['file_uploader'] is not None:
        st.session_state['ctr'] += 1
        print('Uploaded file #%d' % st.session_state['ctr'])

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text







































with st.sidebar:
   selected = option_menu("Meta Data Extraction", ["SignUP","Login Page","Workspaces", "Data Upload"], menu_icon="chevron-down", default_index=0)
 


if selected=="SignUP":
    st.header("Signup")
    new_username = st.text_input("Enter a new username:")
    new_password = st.text_input("Enter a new password:", type='password')
    confirm_password = st.text_input("Confirm password:", type='password')

    if st.button('Signup'):
        if new_password == confirm_password:
            print(new_password)
            if insert_user_credentials(new_username, new_password):
                st.success("Signup successful. You can now log in.")
            else:
                st.error("Error during signup. Please try again.")
        else:
            st.error("Passwords do not match. Please enter matching passwords.")

if selected=="Login Page":
    st.header("Login Page")
    username_input = st.text_input("Username:")
    password_input = st.text_input("Password:", type='password')

    if st.button('Login'):
        # Fetch user data from PostgreSQL
        user_data,user_id = fetch_user_credentials(username_input, password_input)
        print("user------------",user_data,user_id)
        st.session_state.user_id = user_id
        if user_data:
            # st.title(f'Welcome *{user_data["name"]}*')
            st.write("You have successfully Logged In!")
        else:
            st.error('Username/password is incorrect')

if st.session_state.user_id !=0:
    # 'Workspaces' section
    if selected == 'Workspaces':
        workspace_name = st.text_input("Enter Workspace Name")
        # global_headers_input = st.text_input("Enter Global Headers (comma-separated):")
        workspace_created = st.session_state.get('workspace_created', False)
        user_id=st.session_state.user_id 
        print("User_id from Workspace section",user_id)
        # gpt_key = st.text_input("Enter your GPT Key :")
        # OPENAI_API_KEY=f"{gpt_key}"

        # st.session_state.gpt_key=gpt_key
        # update_gptkey(user_id,gpt_key)
        
        if st.button('Create Workspace', key='create_workspace_button'):
            try:
                connection_params = {
                    'host': 'database-1.cmeaoe1g4zcd.ap-south-1.rds.amazonaws.com',
                    'port': '5432',
                    'database': 'postgres',
                    'user': 'postgres',
                    'password': 'postgres'
                }
                connection = psycopg2.connect(**connection_params)
                cursor = connection.cursor()
                cursor.execute("SELECT MAX(id) FROM genpact.workspaces;")
                latest_user_id = cursor.fetchone()[0]
                id = 1 if latest_user_id is None else latest_user_id + 1
                st.session_state.workspace_name = workspace_name
                workspace_created, workspace_id ,user_id= create_workspace(id, workspace_name,user_id)

                # global_headers_input = st.text_input("Enter Global Headers (comma-separated):")
                # print("function before call")
                # print("workspace name ...........",workspace_name)
                # print("global -----------",global_headers_input)
                # # save_global_headers(workspace_name, global_headers_input)
                # print("executed...............")
                        

            except ValueError as e:
                st.warning(str(e))
            finally:
                connection.close()
                print("------------------",workspace_created)
    

        st.write("### Workspace History")
        
        if workspace_created:
           
            st.info("You've created a new workspace. Workspace history selection is disabled.")
            workspace_created=True
            st.session_state.workspace_name = workspace_name
            

        else:
            user_id=st.session_state.user_id
            workspace_history = get_workspace_history(user_id)
            formatted_workspaces = [workspace[0].strip("()").replace(",", "") for workspace in workspace_history]
            selected_workspace = st.selectbox("Select Workspace:", formatted_workspaces, key='workspace_dropdown')

            if selected_workspace:
                user_id=st.session_state.user_id
                # selected_workspace = st.experimental_get_query_params().get('selected', [None])[0]

                connection_params = {
                        'host': 'database-1.cmeaoe1g4zcd.ap-south-1.rds.amazonaws.com',
                        'port': '5432',
                        'database': 'postgres',
                        'user': 'postgres',
                        'password': 'postgres'
                    }
                connection = psycopg2.connect(**connection_params)
                cursor = connection.cursor()
                
                cursor.execute(f"SELECT filename, table_name FROM genpact.workspace_history WHERE user_id={user_id} AND workspace_name='{selected_workspace}';")
                files_and_tables = cursor.fetchall()

                # st.write(f"You selected workspace: {selected_workspace}")


                cursor.execute(f"SELECT DISTINCT filename FROM genpact.workspace_history WHERE user_id={user_id} AND workspace_name='{selected_workspace}';")
                available_files = [row[0] for row in cursor.fetchall()]

                # cursor.execute(f"SELECT gptkey FROM genpact.workspaces WHERE user_id={user_id} AND workspace_name='{selected_workspace}';")

                # gpt_key = cursor.fetchone()
                # OPENAI_API_KEY=f"{gpt_key}"
                # st.session_state.gpt_key=f"{gpt_key}"
                selected_file = st.selectbox("Select File:", available_files, key='file_dropdown')       
                # st.session_state.selected = 'Update Data'
                st.experimental_set_query_params(selected=selected_workspace,selected_file=selected_file)


    if selected == 'Data Upload':
        
        logo_url = "genpactlogo.png"
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(logo_url, width=70)

        with col2:
            st.title("MetaData Extraction")

        filename="None"
        uploaded_file = st.file_uploader("Choose a pdf file", on_change=uploader_callback, type="pdf", key="file_uploader")
        if uploaded_file:
            filename=uploaded_file.name
        gpt_key=st.session_state.gpt_key
        def browse_file():
            if uploaded_file is not None:
                temp_pdf = tempfile.NamedTemporaryFile(delete=False)
                temp_pdf.write(uploaded_file.read())
                policy_uploaded = False

                if not policy_uploaded:
                    # policy_uploaded = upload_policy(temp_pdf)
                    with open(temp_pdf.name, 'rb') as pdf_file:
                        vectorize_and_store_documents(pdf_file.name)

                    #     files = {'pdf_file': (temp_pdf.name, pdf_file, 'application/pdf')}
                    # with open(uploaded_file.name, "wb") as f:
                    #     f.write(uploaded_file.read())
                    # vectorize_and_store_documents(uploaded_file.name)
                    policy_uploaded=True
            st.success("File Uploaded Successfully")
        st.subheader("Select the Model")
        models=["gpt-3.5","gpt-4"]
        selected_model = st.selectbox("Select model:", models)
        if selected_model=='gpt-3.5':
            selected_model="gpt-3.5-turbo-0613"


        
        if selected_model:
            
            st.subheader("MetaData")
            user_id=st.session_state.user_id
            metadata_values = get_metadata_values(user_id)
            # for val in metadata_values:
            #     val.replace("\\","")
            print("///////////////////",metadata_values)
    
            selected_metadata = st.selectbox("Select metadata:", metadata_values)
            formatted_prompt = st.text_area("Enter the attributes you want to extract from the document (comma-separated attributes):")
            metadata_name = st.text_input("Enter the Metadata name:")
            submit_button = st.button("Submit")
            workspace_name=st.session_state.workspace_name
        # formatted_prompt_list = []
        # formatted_prompt_list.append(formatted_prompt)
            
            if (selected_metadata and not formatted_prompt) or submit_button:
                st.text("Selected Metadata:")
                st.write(selected_metadata)
 
                formatted_prompt_list = [selected_metadata]
                formatted_prompt2=f'{selected_metadata}'
            elif (selected_metadata and formatted_prompt) or submit_button:
                st.text("Selected Metadata:")                
                formatted_prompt_list = [selected_metadata]
                formatted_prompt_list.append(formatted_prompt)
                formatted_prompt2=f'{selected_metadata},{formatted_prompt}'
                st.write(formatted_prompt2)
                inser_metadata(workspace_name,filename,user_id,formatted_prompt_list,metadata_name)
            

            elif (formatted_prompt and not selected_metadata) or submit_button:
                print(formatted_prompt)
                st.write(metadata_name)
                st.write(formatted_prompt)
                # formatted_prompt = st.text_area("Enter the attributes you want to extract from the document (comma-separated attributes):")
                formatted_prompt_list = [formatted_prompt]
                formatted_prompt2=f'{formatted_prompt}'
                inser_metadata(workspace_name,filename,user_id,formatted_prompt_list,metadata_name)

            


    # if uploaded_file is not None:

    #     browse_file()
            # if selected_metadata:
            #     formatted_prompt2=f'{selected_metadata}'
            # else:
            #     formatted_prompt2=f'{formatted_prompt}'

            with st.spinner("Processing..."):
                if st.button("Execute"):

                    browse_file()
                    # gpt_key=st.session_state.gpt_key
                    result=call_attributes_api(formatted_prompt2,selected_model)
                    # result_placeholder.result(result)
                    st.session_state.result=result

                    # st.markdown("### Update Attributes Data")
                    # st.markdown("###")
                    # result=st.session_state.result
                    # unique_key = int(time.time())
                    # editable_df = pd.DataFrame(result)
                    # edited_df = st.data_editor(editable_df, key=unique_key, num_rows="dynamic")
                    # modified_df = st.data_editor(st.session_state.edited_df)
                    # st.session_state.edited_df = modified_df

        # modified_df = st.data_editor(st.session_state.edited_df)
        # st.session_state.edited_df = modified_df


        # with st.form("my_form"):
        #     submitted = st.form_submit_button("Save")
        #     if submitted:
        #         on_download_click(st.session_state.edited_df, "output.json")
                    
                    # with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    #     modified_df.to_excel(writer, sheet_name="Sheet1", index=False)

                    # st.download_button(
                    #     label="Download Excel workbook",
                    #     data=buffer.getvalue(),
                    #     file_name="workbook.xlsx",
                    #     mime="application/vnd.ms-excel"
                    # )


            result = st.session_state.result
            print(result)
            if result is not None:
                st.markdown("### Update Attributes Data")
                st.markdown("###")
                unique_key = int(time.time())
                editable_df = pd.DataFrame(result)

                edited_df = st.data_editor(editable_df, key=unique_key, num_rows="dynamic")
                modified_df = st.data_editor(st.session_state.edited_df)
                st.session_state.edited_df = modified_df

                # Create an in-memory Excel workbook

                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    edited_df.to_excel(writer, sheet_name="Sheet1", index=False)

                st.download_button(
                    label="Download Excel workbook",
                    data=buffer.getvalue(),
                    file_name="workbook.xlsx",
                    mime="application/vnd.ms-excel"
                )
                buffer.close()

