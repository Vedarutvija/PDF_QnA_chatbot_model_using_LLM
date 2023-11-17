from PyPDF2 import PdfReader
import streamlit as st
import re
from streamlit_chat import message
from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

st.title("Question Answering Chatbot")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

index_name = st.text_input("Enter Index Name:")

if st.button("Submit Document"):
    if uploaded_file and index_name:

        document_content = ""
        for page in PdfReader(uploaded_file).pages:
            document_content += page.extract_text()
        document_content = re.sub(r'[^a-zA-Z0-9\s]', '', document_content)

        es_document = {"content": document_content}
        es.index(index=index_name, body=es_document)

        st.success(f"Document from PDF submitted to index: {index_name}")
    else:
        st.warning("Please upload a PDF file and enter an index name.")

st.sidebar.title("Chatbot")

user_query = st.sidebar.text_input("Ask me anything:")

if st.sidebar.button("Submit Query"):
    if user_query:

        search_results = es.search(index=index_name, body={"query": {"match": {"content": user_query}}})

        if search_results['hits']['total']['value'] > 0:
            st.sidebar.success("Found matching documents:")
            for hit in search_results['hits']['hits']:
                st.sidebar.write(hit['_source']['content'])
        else:
            st.sidebar.warning("No matching documents found.")
    else:
        st.sidebar.warning("Please enter a query.")