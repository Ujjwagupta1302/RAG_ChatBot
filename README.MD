# 📄 RAG Chatbot Application

This project presents a Retrieval Augmented Generation (RAG) powered chatbot application designed to provide contextual answers based on user-provided documents and web content. Built with a focus on conversational fluidity and information retrieval, this chatbot leverages the power of Large Language Models (LLMs) and vector databases to deliver accurate and relevant responses.

## ✨ Features

* **Multi-Document Support**: Upload up to 5 documents (e.g., PDF, TXT) and enable the chatbot to converse based on their content.
* **Web Content Integration**: Paste a URI (URL) to allow the chatbot to ingest data from web pages using `WebBaseLoader` and conduct conversations based on that online content.
* **Context-Aware Chat**: Utilizes previous chat history to maintain context during conversations, providing a more natural and coherent user experience.
* **Retrieval Augmented Generation (RAG)**: Integrates document retrieval with LLM generation to ensure answers are grounded in the provided information, minimizing hallucinations.
* **Streamlit UI**: Features an intuitive and interactive user interface developed with Streamlit, making the application easy to use and deploy.

## 🛠️ Technologies Used

The core components and libraries used in this project include:

* **LangChain**: For orchestrating the LLM, retrieval, and conversational chains.
* **Streamlit**: For building the interactive web application interface.
* **Chroma DB**: As the vector database for storing and retrieving document embeddings.
* **Embedding Function**: To convert text into numerical vector embeddings for efficient similarity search.
* **Large Language Models (LLMs)**: Powering the conversational capabilities and answer generation.

## 🚀 Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure your `requirements.txt` file lists all your dependencies like `streamlit`, `langchain`, `chromadb`, etc.)
4.  **Set up environment variables**:
    If your LLM or other services require API keys, create a `.env` file in the root directory and add your keys:
    ```
    OPENAI_API_KEY="your_openai_api_key" # Or your specific LLM API key
    ```
    (You might need to install `python-dotenv` if not already included in `requirements.txt`)

## 💡 Usage

1.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```
2.  **Interact with the Chatbot**:
    * The Streamlit application will open in your web browser.
    * Use the interface to upload up to 5 documents or paste a URI.
    * Start chatting with the bot, and it will respond based on the provided context and your conversation history.

## 🗺️ Deployment

The application is designed to be deployed using Streamlit. You can deploy it to Streamlit Cloud or other platforms that support Streamlit applications.


## 🚀 Future Improvements

This project is continuously evolving, and here are some areas targeted for future enhancements:

* **Support for More Document Types**: Expand document processing capabilities to include formats like `.pptx`, `.csv`, etc.
* **Enhanced UI/UX**: Implement more interactive elements, progress indicators during document loading/processing, and refined chat interface.
* **User Feedback Mechanism**: Allow users to provide feedback on chatbot responses to help in continuous improvement and model fine-tuning.
* **Chrome Extension for Web Content**: Develop a browser extension (e.g., for Chrome) that allows users to directly converse with the content of the currently viewed website, integrating seamlessly with the chatbot's capabilities.