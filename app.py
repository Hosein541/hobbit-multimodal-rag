import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
import sys 
import os 
from pipelines.ingestion import create_db
from chains.chat import get_answers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOOK_DIR, METADATA_DIR, IMAGE_DIR, CHROMA_DIR
from pathlib import Path
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

@st.cache_resource
def load_resources(llm):

    image_retriever, text_retriever = create_db(llm)

    return image_retriever, text_retriever

with st.sidebar:
    st.header("API Keys")
    gemini_key = st.text_input("Gemini API Key", type="password")
    model_selection = st.selectbox("Model Selection", ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.1-pro-preview"])
    start_button = st.button("🚀 Generate Initial Analysis", type="primary")

if gemini_key and start_button:
    st.session_state.llm = ChatGoogleGenerativeAI(model=model_selection, temperature=0.6, google_api_key=gemini_key)

    image_retriever, text_retriever = load_resources(st.session_state.llm)
    st.session_state.img_ret = image_retriever
    st.session_state.text_ret = text_retriever


# =========================================
# Initialize Retrievers
# =========================================



# embed_llm = OllamaLLM(model="gemma3:4b", temperature=0)

# result = text_retriever.invoke("hello")
# st.markdown(result)

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
    # llm = st.session_state.llm
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

            results = get_answers(
                llm=st.session_state.llm,
                question=question,
                image_vectorstore=st.session_state.img_ret,
                text_retriever= st.session_state.text_ret
            )
            answer = results["text_answer"]
            image_paths = results["img_result"]
            # image_paths = st.session_state.img_ret.invoke("gandalf")
            # answer = "hello "
            # image_paths = IMAGE_DIR / "page_35_img_0.jpeg"
        st.markdown(answer)
        # st.markdown(image_paths)
        # st.markdown(text_ret)

        if image_paths:

            st.divider()
            st.subheader("Retrieved Images")

            # cols = st.columns(2)
            path = []
            for i in image_paths:
                # st.markdown(i.metadata["image_id"])
                path.append(i.metadata["image_id"])

            num_images = len(path)  # فرض کنید path لیستی از شناسه‌های تصاویر است
            cols = st.columns(num_images)  # به تعداد تصاویر ستون بسازید

            for idx, img_id in enumerate(path):
                with cols[idx]:
                    st.image(Path(f"data/images/{img_id}.jpeg"))
            # try:
            #     with cols[0]:
            #         st.image(Path(f"data/images/{path[0]}.jpeg"))
            #     with cols[1]:
            #         st.image(Path(f"data/images/{path[1]}.jpeg"))
            # except Exception as e:
            #     st.warning(f"خطا در نمایش تصاویر: {e}")
            # with cols[10 % len(cols)]:
            #     try:
            #         st.image(
            #             Path(f"data/images/{path[0]}.jpeg"),
            #                             #   page_147_img_0
            #             # Path("data/images/page_147_img_0.jpeg"),
            #             # use_container_width=True
            #         )
            #         st.image(
            #             Path(f"data/images/{path[1]}.jpeg"),
            #                             #   page_147_img_0
            #             # Path("data/images/page_147_img_0.jpeg"),
            #             # use_container_width=True
            #         )
            #     except:
            #         pass 

            # for idx, image_path in enumerate(image_paths):

            #     with cols[idx % len(cols)]:

            #         try:
            #             st.image(
            #                 image_path,
            #                 use_container_width=True
            #             )
            #         except Exception as e:

            #             st.warning(
            #                 f"Cannot load image:\n{image_path}"
            #             )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
