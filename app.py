import streamlit as st
import os
import dotenv
import uuid

# check if it's linux so it works on Streamlit Cloud
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db, 
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("Please set your GOOGLE_API_KEY environment variable.")

# Only Gemini model choices
MODELS = [
    "google/gemini-2.0-flash",  # or "google/gemini-1.5-pro" if you prefer
]


st.set_page_config(
    page_title="RAG LLM app", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)


# --- Header ---
st.html("""<h2 style="text-align: center;">📚🔍 Would You like to talk to your Documents? </i> 🤖💬</h2>""")


# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
]


# --- Side Bar LLM API Tokens ---
with st.sidebar:
    if "GOOGLE_API_KEY" not in os.environ:
        default_gemini_key = os.getenv("GOOGLE_API_KEY") or ""  # Safe fallback for local dev
        with st.popover("🔐 Gemini"):
            gemini_api_key = st.text_input(
                "Enter your Gemini API Key (https://makersuite.google.com/app/apikey)", 
                value=default_gemini_key, 
                type="password",
                key="gemini_api_key",
            )
        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
            st.session_state.gemini_api_key = gemini_api_key
    else:
        st.session_state.gemini_api_key = os.getenv("GOOGLE_API_KEY")



# --- Main Content ---
# Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
missing_gemini = (
    "GOOGLE_API_KEY" not in os.environ
    or os.environ["GOOGLE_API_KEY"] == ""
    or st.session_state.get("gemini_api_key", "") == ""
)

if missing_gemini:
    st.write("#")
    st.warning("⬅️ Please enter your Gemini API Key in the sidebar to continue...")


else:
    # Sidebar
    with st.sidebar:
        st.divider()
        models = []
        for model in MODELS:
            if "gemini" in model:
                models.append(model)


        st.selectbox(
            "🤖 Select a Model", 
            options=models,
            key="model",
        )

        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle(
                "Use RAG", 
                value=is_vector_db_loaded, 
                key="use_rag", 
                disabled=not is_vector_db_loaded,
            )

        with cols0[1]:
            st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

        st.header("RAG Sources:")
            
        # File upload input for RAG with documents
        st.file_uploader(
            "📄 Upload a document (Upto 5)", 
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs",
        )

        # URL input for RAG with websites
        st.text_input(
            "🌐 Introduce a URL", 
            placeholder="https://example.com",
            on_change=load_url_to_db,
            key="rag_url",
        )

        with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
            st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])

    
    # Main chat app
    model_provider = st.session_state.model.split("/")[0]
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm_stream = ChatGoogleGenerativeAI(
        model=st.session_state.model.split("/")[-1],  # e.g., "gemini-1.5-flash"
        temperature=0.3,
        google_api_key=st.session_state.gemini_api_key,
    )


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

            if not st.session_state.use_rag:
                st.write_stream(stream_llm_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_rag_response(llm_stream, messages))


with st.sidebar:
    st.divider()

    

    