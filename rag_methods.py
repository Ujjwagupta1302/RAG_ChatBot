import streamlit as st
import os
from dotenv import load_dotenv
from time import time

# Langchain loaders and tools
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

# Langchain chains and prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough

# Gemini 2.0 Flash via LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Constants
DB_DOCS_LIMIT = 10
os.environ["USER_AGENT"] = "myagent"

# Load Gemini Flash Model
def get_gemini_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash-latest",
        temperature=0.3,
        convert_system_message_to_human=True,
        verbose=True,
    )


def stream_llm_response(llm_stream, messages):
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response_message})


def load_doc_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Unsupported document type: {doc_file.type}")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)
                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                    finally:
                        os.remove(file_path)
                else:
                    st.error(f"Maximum document limit reached ({DB_DOCS_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast("Document(s) loaded successfully.", icon="✅")


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)
                except Exception as e:
                    st.error(f"Error loading URL {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast("URL loaded successfully.", icon="✅")
            else:
                st.error("Maximum document limit reached.")


def _split_and_load_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    splits = text_splitter.split_documents(documents)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(splits)
    else:
        st.session_state.vector_db.add_documents(splits)


def initialize_vector_db(documents):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id']
    )

    # Cleanup old collections
    chroma_client = vector_db._client
    collection_names = sorted([c.name for c in chroma_client.list_collections()])
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db


def _get_context_retriever_chain(vector_db, llm) :
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    contextualize_q_system_prompt = """
        Given a chat history and the latest user question
        which might reference context in the chat history,
        formulate a standalone question which can be understood
        without the chat history. Do NOT answer the question,
        just reformulate it if needed and otherwise return it as is.
        Do not consider the question if it is not present in the context which is provided to you. 
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    return history_aware_retriever


def get_conversational_rag_chain(llm) :
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use only the following context to answer the user's question. If you cannot find the answer from the given context, say 'Sorry, not available'"),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever_chain, question_answer_chain)

    return rag_chain


def stream_llm_rag_response(llm_stream, messages):
    rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    for chunk in rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})
