import streamlit as st
import pandas as pd
import requests
import tempfile
import time

st.title("MetaData Extraction")

uploaded_file = st.file_uploader("Choose a pdf file", type="pdf")

policy_uploaded = False
table_shown = False 

editable_df = pd.DataFrame()
if uploaded_file is not None:
    print(3)

    temp_pdf = tempfile.NamedTemporaryFile(delete=False)
    temp_pdf.write(uploaded_file.read())
    st.markdown("###")

st.subheader("MetaData")
formatted_prompt = st.text_area("Enter the attributes you want to extract from the document (comma-separated attributes):")

with st.spinner("Processing..."):
    if st.button("Execute"):
        if not policy_uploaded:
            try:
                with open(temp_pdf.name, 'rb') as pdf_file:
                    files = {'pdf_file': (temp_pdf.name, pdf_file, 'application/pdf')}
                    response_up = requests.post("http://localhost:8000/upload-policy/", files=files)

                if response_up.status_code == 200:
                    st.subheader("Processed Policy Upload Data")
                    policy_uploaded = True
                else:
                    st.error(f"Error processing file for policy upload. Status code: {response_up.status_code}")

            except requests.exceptions.RequestException as policy_error:
                st.error(f"Error in policy upload request: {policy_error}")
            st.markdown("###")

        if formatted_prompt:
            try:
                data = {'formatted_prompt': formatted_prompt}
                response_att = requests.post("http://localhost:8000/attributes/", json=data, headers={'Content-Type': 'application/json'})
                if response_att.status_code == 200:
                    st.markdown("### Processed Attributes Data")
                    unique_key = int(time.time())
                    editable_df = pd.DataFrame(response_att.json())
                    edited_df = st.data_editor(editable_df, key=unique_key, num_rows="dynamic")
                    st.markdown("###")
                    table_shown = True 
                    st.stop()


                else:
                    st.error(f"Error processing attributes. Status code: {response_att.status_code}")
                st.markdown("###")
            except requests.exceptions.RequestException as attribute_error:
                st.error(f"Error in attributes request: {attribute_error}")

        st.markdown("### Update Attributes Data")
        st.write(edited_df)
        st.markdown("###")
        st.markdown("###")
        if 'temp_pdf' in locals():
            temp_pdf.close()
        st.markdown("###")

if not table_shown:
    edited_df = st.data_editor(editable_df, num_rows="dynamic")
    st.stop()
