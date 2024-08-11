# Streamlit PDF Chat Application

This Streamlit application provides a chat interface for interacting with PDF documents using a Retrieval-Augmented Generation (RAG) approach with LlamaIndex. Users can upload PDF files or provide PDF URLs, and the application will process and query the content in real-time, with embeddings stored temporarily in memory.

## Features

- **PDF Upload**: Users can upload PDF files directly through the Streamlit interface.
- **PDF URL Input**: Users can provide URLs to PDF files hosted online.
- **Chat Interface**: Provides a chat interface for querying the content of uploaded or linked PDFs.
- **In-Memory Embeddings**: Uses in-memory storage for embeddings, eliminating the need for a persistent vector database.

## Installation

To set up this project, follow these steps:

1. **Clone the Repository:**

    ```bash
    git clone <chatwithpdf-llamaindex-streamlit-application>
    cd <repository-directory>
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**

    Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. **Hugging Face Authentication:**

    Ensure you have a Hugging Face token. You can obtain one by creating an account on [Hugging Face](https://huggingface.co/) and generating a token from the settings page.

## Configuration

Update the following configurations in your Streamlit application script:

- **Hugging Face Token:**

    Replace `'YOUR_TOKEN'` with your actual Hugging Face token:

    ```python
    token='YOUR_TOKEN'
    ```

- **Model Settings:**

    Ensure the settings for the model are correctly configured:

    ```python
    from llama_index.llms.huggingface import HuggingFaceInferenceAPI
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.llm = HuggingFaceInferenceAPI(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
        context_window=3900,
        token='YOUR_TOKEN',
        max_new_tokens=1000,
        generate_kwargs={"temperature": 0},
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 64
    ```

## Usage

1. **Run the Streamlit Application:**

    Start the Streamlit server with the following command:

    ```bash
    streamlit run app.py
    ```

    This will open the application in your default web browser.

2. **Upload PDF or Provide URL:**

    - **Upload PDF**: Use the file upload widget to upload a PDF file from your local system.
    - **PDF URL**: Enter the URL of a PDF file hosted online.

3. **Interact with the PDF:**

    After uploading or providing a PDF URL, use the chat interface to query the content of the PDF. The system will process your query and respond based on the content of the uploaded PDF.
