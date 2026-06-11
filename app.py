import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
import sys 
import os 
from pipelines.ingestion import create_db
from chains.chat import get_answers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOOK_DIR, METADATA_DIR, IMAGE_DIR, CHROMA_DIR

# =========================================
# Page Config
# =========================================

st.set_page_config(
    page_title="Multimodal Hobbit RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multimodal Hobbit RAG")
st.caption(
    "Text + Image Retrieval powered by Ollama, ChromaDB and LangChain"
)


# =========================================
# Initialize Retrievers
# =========================================

@st.cache_resource
def load_resources(llm):

    text_retriever, image_retriever = create_db(llm)

    return text_retriever, image_retriever

embed_llm = OllamaLLM(model="gemma3:4b", temperature=0)
text_retriever, image_retriever = load_resources(embed_llm)
st.session_state.img_ret = image_retriever
st.session_state.text_ret = text_retriever


# =========================================
# Chat History
# =========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================================
# User Input
# =========================================

question = st.chat_input(
    "Ask something about The Hobbit..."
)


if question:
    # llm = OllamaLLM(model="gemma3:4b", temperature=0)
    llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            google_api_key="AIzaSyCpKQdGGbxKClS-2YcCHIF1YVPYt5aOSD4",
            temperature=0.1,
        )
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            # results = get_answers(
            #     llm=llm,
            #     question=question,
            #     image_retriever=st.session_state.img_ret,
            #     text_retriever= st.session_state.text_ret
            # )
            # answer = results["text_answer"]
            # image_paths = results["img_result"]
            image_paths = st.session_state.img_ret.invoke("gandalf")
            answer = "hello "
            # image_paths = IMAGE_DIR / "page_35_img_0.jpeg"
        st.markdown(answer)

        if image_paths:

            st.divider()
            st.subheader("Retrieved Images")

            cols = st.columns(
                min(3, 4)
            )

            for idx, image_path in enumerate(image_paths):

                with cols[idx % len(cols)]:

                    try:
                        st.image(
                            image_path,
                            use_container_width=True
                        )
                    except Exception as e:

                        st.warning(
                            f"Cannot load image:\n{image_path}"
                        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )