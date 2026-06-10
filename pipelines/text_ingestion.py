import re
import sys 
import os 
import json 
import fitz
import base64
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOOK_DIR, METADATA_DIR, IMAGE_DIR, CHROMA_DIR

def split():
    pdf = fitz.open(f"{BOOK_DIR}/The Hobbit.pdf")


    pages = []

    for page_num in range(len(pdf)):
        pages.append({
            "page": page_num + 1,
            "text": pdf[page_num].get_text()
        })

    with open(
        "image_metadata_final.json",
        "r",
        encoding="utf-8"
    ) as f:
        image_data = json.load(f)





    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )


    page_to_images = {}

    for img in image_data:

        page = img["page"]

        if page not in page_to_images:
            page_to_images[page] = []

        page_to_images[page].append(
            img["image_id"]
        )


    docs = []

    for page in pages:

        chunks = splitter.split_text(
            page["text"]
        )


        for idx, chunk in enumerate(chunks):

            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "page": page["page"],
                        "chunk": idx,
                        "source": "text",
                    }
                )
            )

    return docs


def build_text_collection():
    
    db_name = "chroma_db"
    persist_directory = os.path.join(
        CHROMA_DIR,
        db_name
    )

    embeddings = OllamaEmbeddings(
        model="embeddinggemma"
    )
    # -------------------------------------
    # Load Existing DB
    # -------------------------------------
    
    if os.path.exists(persist_directory):

        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name = "hobbit_text"
        )
        print(f"vector store is loaded {db_name}")

    else:

        docs = split()
        print(f"vector store creating \t{db_name}")
        print(
            f"Total documents: {len(docs)}"
        )
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name = "hobbit_text"
        )


    # docs = split()

    # embeddings = OllamaEmbeddings(
    #     model="embeddinggemma"
    # )

    # vectorstore = Chroma.from_documents(
    #     documents=docs,
    #     embedding=embeddings,
    #     collection_name="hobbit_text",
    #     persist_directory="/content/drive/MyDrive/hobbit/chroma_db"
    # )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    return retriever