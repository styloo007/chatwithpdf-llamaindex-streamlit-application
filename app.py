import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from youtube_transcript_api import YouTubeTranscriptApi
import shutil
import os
import time
from datetime import datetime
import requests

icons = {"assistant": "robot.png", "user": "man-kddi.png"}

# Configure the Llama index settings
Settings.llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
    context_window=3900,
    token=os.getenv("HF_TOKEN"),
    max_new_tokens=1000,
    generate_kwargs={"temperature": 0},
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5"
)

# Define the directory for persistent storage and data
PERSIST_DIR = "./db"
DATA_DIR = "data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def data_ingestion():
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

def remove_old_files():
    # Specify the directory path you want to clear
    directory_path = "data"

    # Remove all files and subdirectories in the specified directory
    shutil.rmtree(directory_path)

    # Recreate an empty directory if needed
    os.makedirs(directory_path)

def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]
       
        return transcript

    except Exception as e:
        st.error(e)

def download_pdf_from_url(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        pdf_path = os.path.join(DATA_DIR, "downloaded_pdf.pdf")
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(response.content)
        return pdf_path
    else:
        st.error("Failed to download PDF from the provided URL.")
        return None

def handle_query(query):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    chat_text_qa_msgs = [
        (
            "user",
            """You are a QA based chatbot, developed by Shashank Agasimani an AI/ML Developer. Your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, you only say the user to 'Please ask a questions within the context of the document'.
            Context:
            {context_str}
            Question:
            {query_str}
            """
        )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    query_engine = index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)
    
    if hasattr(answer, 'response'):
        return answer.response
    elif isinstance(answer, dict) and 'response' in answer:
        return answer['response']
    else:
        return "Sorry, I couldn't find an answer."

def streamer(text):
    for i in text:
        yield i
        time.sleep(0.001)


# Streamlit app initialization
st.title("Chat with your PDF 📄 a POC")

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": 'Hello! Upload a PDF/Youtube Video link and ask me anything about the content.'}]

for message in st.session_state.messages:
    with st.chat_message(message['role'], avatar=icons[message['role']]):
        st.write(message['content'])

with st.sidebar:
    st.title("Menu:")
    uploaded_file = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button")
    video_url = st.text_input("Enter YouTube Video Link: ")
    pdf_url = st.text_input("Enter PDF URL: ")
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            if len(os.listdir("data")) != 0:
                remove_old_files()
            
            if uploaded_file:
                filepath = f"data/{uploaded_file.name}"
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        
            if video_url:
                extracted_text = extract_transcript_details(video_url)
                with open("data/saved_text.txt", "w") as file:
                    file.write(extracted_text)
            
            if pdf_url:
                download_pdf_from_url(pdf_url)
            
            data_ingestion()  # Process PDF every time new file is uploaded
            st.success("Done")

user_prompt = st.chat_input("Ask me anything about the content of the PDF or YouTube video:")

if user_prompt and (uploaded_file or video_url or pdf_url):
    st.session_state.messages.append({'role': 'user', "content": user_prompt})
    with st.chat_message("user", avatar="man-kddi.png"):
        st.write(user_prompt)

    # Trigger assistant's response retrieval and update UI
    with st.spinner("Thinking..."):
        response = handle_query(user_prompt)
    with st.chat_message("assistant", avatar="robot.png"):
        st.write_stream(streamer(response))
    st.session_state.messages.append({'role': 'assistant', "content": response})
