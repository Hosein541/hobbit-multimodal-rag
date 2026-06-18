import re
import sys 
import os 
import json 
import fitz
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOOK_DIR, METADATA_DIR, IMAGE_DIR, CHROMA_DIR


def split():

    pdf = fitz.open(f"{BOOK_DIR}/The Hobbit.pdf")

    page_texts = []
    page_starts = []  
    combined_text = ""

    for page_num in range(len(pdf)):
        text = pdf[page_num].get_text()
        page_texts.append(text)
        page_starts.append(len(combined_text))  
        combined_text += text  

    with open(f"{METADATA_DIR}/image_metadata_final.json", "r", encoding="utf-8") as f:
        image_data = json.load(f)

    page_to_images = {}
    for img in image_data:
        page = img["page"]
        page_to_images.setdefault(page, []).append(img["image_id"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(combined_text)

    docs = []
    for idx, chunk in enumerate(chunks):

        start_pos = combined_text.find(chunk)
        if start_pos == -1:

            start_pos = 0

        page_num = 1  
        for i, p_start in enumerate(page_starts):
            if start_pos >= p_start:
                page_num = i + 1 
            else:
                break

        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "page": page_num,        
                    "chunk": idx,
                    "source": "text",
                   
                }
            )
        )

    return docs

def get_vectorstore(persist_directory, collection_name, embeddings):
    docs = split()  

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
        print(f"📁 Folder is created, creating the vectorstore")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print(f"✅ collection with {len(docs)} document created.")
        return vectorstore


    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    try:
        
        doc_count = vectorstore._collection.count()
    except Exception:

        doc_count = len(vectorstore.get()['ids'])

    if doc_count == 0:
        print(f"⚠️ collection '{collection_name}' is empty, creating it...")
        
        try:
            vectorstore._client.delete_collection(collection_name)
        except:
            pass

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print(f"✅ collection with {len(docs)} document created again.")
    else:
        print(f"✅ Collection successfully imported, the number of documents: {doc_count}")

    return vectorstore



def build_text_collection():
    
    db_name = "chroma_db"
    persist_directory = os.path.join(
        CHROMA_DIR,
        db_name
    )

    collection_name = "hobbit_text"

    embeddings = OllamaEmbeddings(
        model="embeddinggemma",
        base_url="http://localhost:11434",  # default, change if needed

    )

    vectorstore = get_vectorstore(persist_directory, collection_name, embeddings)

    # Access the internal Chroma client
    client = vectorstore._client 

    # Now use the same logic as above
    collections = client.list_collections()
    for coll in collections:
        print("vectorstore information")
        print(coll.name, coll.count())

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )


    return retriever