import os
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from collections import deque
from keras.models import Sequential
from keras.layers import Dense

def initialize_session_state():
    """Initialize session state variables."""
    pdfs_processed = False
    knowledge_base = None
    messages = deque(maxlen=6)
    all_messages = []
    return pdfs_processed, knowledge_base, messages, all_messages

def build_neural_network(input_size):
    """Build a simple neural network model."""
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def process_pdfs(pdfs):
    """Extract text from PDFs and create a knowledge base."""
    text = ''
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base, chunks, embeddings

def answer_question(knowledge_base, question):
    """Generate an answer to a question using the knowledge base."""
    docs = knowledge_base.similarity_search(question)

    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
        print(cb)

    return response
def display_chat(messages):
    for i, msg in enumerate(messages):
        print(msg['message'], "(You)" if msg['is_user'] else "(Bot)")

def main():
    pdfs_processed, knowledge_base, messages, all_messages = initialize_session_state()

    openai_key = input('Enter your OpenAI API key (hidden): ')
    os.environ["OPENAI_API_KEY"] = openai_key

    pdfs = input('Enter PDF file paths separated by space: ').split()

    if pdfs and not pdfs_processed:
        knowledge_base, chunks, embeddings = process_pdfs(pdfs)
        pdfs_processed = True

        # Convert chunks to a numerical representation for the neural network
        X_train = np.array([embeddings.embed_query(chunk) for chunk in chunks])
        y_train = np.random.randint(2, size=len(chunks))  # Dummy labels for now, replace with actual labels

        # Build and train the neural network
        model = build_neural_network(X_train.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=32)  # Adjust epochs and batch_size as needed

    while pdfs_processed:
        user_question = input('Ask a question about your PDF (or type "exit" to quit): ')

        if user_question.lower() == 'exit':
            break

        response = answer_question(knowledge_base, user_question)
        messages.append({'message': user_question, 'is_user': True})
        messages.append({'message': response, 'is_user': False})
        all_messages.extend(messages)
        display_chat(messages)


if __name__ == '__main__':
    main()
