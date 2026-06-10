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
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOOK_DIR, METADATA_DIR, IMAGE_DIR, CHROMA_DIR









def get_page_text(pdf, page_idx):
    if page_idx < 0 or page_idx >= len(pdf):
        return ""

    try:
        return pdf[page_idx].get_text().strip()
    except Exception:
        return ""

def get_metadata():

    pdf = fitz.open(f"{BOOK_DIR}/The Hobbit.pdf")

    WINDOW_SIZE = 1

    IMAGE_DIR.mkdir(exist_ok=True)
    METADATA_DIR.mkdir(exist_ok=True)

    results = []
    for page_num in range(len(pdf)):

        page = pdf[page_num]

        images = page.get_images(full=True)

        for img_idx, img in enumerate(images):

            xref = img[0]

            try:
                image_data = pdf.extract_image(xref)
            except Exception:
                continue

            ext = image_data.get("ext", "png")

            image_name = f"page_{page_num + 1}_img_{img_idx}.{ext}"

            image_path = IMAGE_DIR / image_name

            with open(image_path, "wb") as f:
                f.write(image_data["image"])

            context_pages = []
            context_parts = []

            for offset in range(-WINDOW_SIZE, WINDOW_SIZE + 1):

                idx = page_num + offset

                if idx < 0 or idx >= len(pdf):
                    continue

                page_text = get_page_text(pdf, idx)

                context_pages.append(idx + 1)

                if page_text:
                    context_parts.append(
                        f"[PAGE {idx + 1}]\n{page_text}"
                    )

            context_text = "\n\n".join(context_parts)

            results.append(
                {
                    "image_id": f"page_{page_num + 1}_img_{img_idx}",
                    "page": page_num + 1,
                    "image_path": str(image_path),

                    "context_pages": context_pages,

                    "context_text": context_text,

                    "caption": None,
                    "characters": [],
                    "location": None
                }
            )

    with open(
        f"{METADATA_DIR}/image_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Done. Extracted {len(results)} images.")



def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    

def format_input(item):
    return {
        "context": item["context_text"][:4000],
        "image": encode_image(item["image_path"]),
    }


def build_chain(llm, prompt):
    return (
        RunnableLambda(format_input)
        | RunnablePassthrough.assign(
            result = (
                prompt
                | llm
            )
        )
    )


def parse_vlm_json(text):

    try:
        return json.loads(text)

    except Exception:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:

        try:
            return json.loads(
                match.group()
            )

        except Exception:
            pass

    return None


def extract_caption(llm):
    with open("image_metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)




    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a multimodal vision-language assistant. "
         "You analyze book illustrations from The Hobbit."),

        ("human",
         """
    Here is the story context:

    {context}

    Describe the image and return STRICT JSON with keys:
    - caption
    - characters (list)
    - location
    - actions
    - objects
    """)
    ])

    chain = build_chain(llm, prompt)
    i = 0
    for item in data:
      if item["page"] > 13 :
        print(f"item number\t{item['page']}")

        try:
            out = chain.invoke(item)
            print(f"item number\t{i}")
            print(f"output\t: {out["result"]}")
            print("----------------------")
            i += 1
            item["vlm_raw"] = out["result"]

        except Exception as e:
            item["vlm_raw"] = str(e)


        parsed = parse_vlm_json(
            item.get("vlm_raw", "")
        )
    
        if not parsed:
            continue
        
        item["caption"] = parsed.get(
            "caption"
        )
    
        item["characters"] = parsed.get(
            "characters",
            []
        )
    
        item["location"] = parsed.get(
            "location"
        )
    
        item["actions"] = parsed.get(
            "actions",
            []
        )
    
        item["objects"] = parsed.get(
            "objects",
            []
        )
    
    with open(
        f"{METADATA_DIR}/image_metadata_final.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def ingestion():
    
    with open(
        "image_metadata_final.json",
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    docs = []

    for item in data:

        content = f"""
    Caption:
    {item.get("caption","")}

    Characters:
    {", ".join(item.get("characters",[]))}

    Location:
    {item.get("location","")}

    Actions:
    {", ".join(item.get("actions",[]))}

    Objects:
    {", ".join(item.get("objects",[]))}
    """

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "image_id": item["image_id"],
                    "page": item["page"],
                    "image_path": item["image_path"],
                    "source": "image"
                }
            )
        )

    return docs



def build_image_collection(llm):

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
            collection_name = "hobbit_images"
        )
        print(f"vector store is loaded {db_name}")

    else:
        get_metadata()

        extract_caption(llm)

        docs = ingestion()
        print(f"vector store creating \t{db_name}")
        print(
            f"Total documents: {len(docs)}"
        )
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name = "hobbit_images"
        )

    
    # get_metadata()
# 
    # extract_caption(llm)
# 
    # docs = ingestion()
# 
    # vectorstore = Chroma.from_documents(
        # documents=docs,
        # embedding=embeddings,
        # collection_name="hobbit_images",
        # persist_directory=f"{CHROMA_DIR}/chroma_db"
    # )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 1
        }
    )

    return retriever

