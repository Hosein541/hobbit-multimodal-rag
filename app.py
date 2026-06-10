import streamlit as st
from langchain_ollama import OllamaLLM

from pipelines.ingestion import create_db
from chains.chat import get_answers


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
def load_resources():
    llm = OllamaLLM(model="gemma3:4b", temperature=0)

    text_retriever, image_retriever = create_db(llm)

    return text_retriever, image_retriever


text_retriever, image_retriever = load_resources()


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

            answer, image_paths = get_answers(
                llm=llm,
                question=question
            )

        st.markdown(answer)

        if image_paths:

            st.divider()
            st.subheader("Retrieved Images")

            cols = st.columns(
                min(3, len(image_paths))
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