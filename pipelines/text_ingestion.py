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

# def split():
#     pdf = fitz.open(f"{BOOK_DIR}/The Hobbit.pdf")


#     pages = []

#     for page_num in range(len(pdf)):
#         pages.append({
#             "page": page_num + 1,
#             "text": pdf[page_num].get_text()
#         })

#     with open(
#         f"{METADATA_DIR}/image_metadata_final.json",
#         "r",
#         encoding="utf-8"
#     ) as f:
#         image_data = json.load(f)





#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )


#     page_to_images = {}

#     for img in image_data:

#         page = img["page"]

#         if page not in page_to_images:
#             page_to_images[page] = []

#         page_to_images[page].append(
#             img["image_id"]
#         )


#     docs = []

#     for page in pages:

#         chunks = splitter.split_text(
#             page["text"]
#         )


#         for idx, chunk in enumerate(chunks):

#             docs.append(
#                 Document(
#                     page_content=chunk,
#                     metadata={
#                         "page": page["page"],
#                         "chunk": idx,
#                         "source": "text",
#                     }
#                 )
#             )

#     return docs

def split():

    pdf = fitz.open(f"{BOOK_DIR}/The Hobbit.pdf")

    # 1. خواندن متن تمام صفحات و ذخیره‌ی موقعیت هر صفحه
    page_texts = []
    page_starts = []  # ایندکس شروع هر صفحه در متن ترکیبی
    combined_text = ""

    for page_num in range(len(pdf)):
        text = pdf[page_num].get_text()
        page_texts.append(text)
        page_starts.append(len(combined_text))  # موقعیت شروع این صفحه
        combined_text += text  # چسباندن بدون جداکننده‌ی خاص

    # 2. خواندن متادیتای تصاویر (برای استفاده‌ی بعدی)
    with open(f"{METADATA_DIR}/image_metadata_final.json", "r", encoding="utf-8") as f:
        image_data = json.load(f)

    page_to_images = {}
    for img in image_data:
        page = img["page"]
        page_to_images.setdefault(page, []).append(img["image_id"])

    # 3. تکه‌تکه کردن کل متن (بدون در نظر گرفتن مرز صفحات)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(combined_text)

    # 4. ساخت Document برای هر تکه، با متادیتای صفحه‌ی شروع
    docs = []
    for idx, chunk in enumerate(chunks):
        # پیدا کردن ایندکس شروع این تکه در کل متن
        # (چون splitter ممکن است با overlap کار کند، باید موقعیت دقیق را پیدا کنیم)
        # روش ساده: جستجوی اولین occurrence از chunk در combined_text (با توجه به overlap ممکن است تکراری باشد)
        # بهتر: از متد splitter که موقعیت‌ها را برمی‌گرداند استفاده کنیم، ولی ساده‌ترین راه:
        start_pos = combined_text.find(chunk)
        if start_pos == -1:
            # اگر پیدا نشد (به دلیل overlap یا تغییرات)، از روش تقریبی استفاده می‌کنیم
            # اما معمولاً find کار می‌کند
            start_pos = 0

        # پیدا کردن صفحه‌ای که این تکه در آن شروع می‌شود
        page_num = 1  # پیش‌فرض
        for i, p_start in enumerate(page_starts):
            if start_pos >= p_start:
                page_num = i + 1  # شماره صفحه (۱-اندیس)
            else:
                break

        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "page": page_num,          # شماره‌ی صفحه‌ی شروع
                    "chunk": idx,
                    "source": "text",
                    # (اختیاری) می‌توانید محدوده‌ی صفحه‌ها را هم ذخیره کنید
                    # "page_range": f"{page_num}-..."
                }
            )
        )

    return docs

def get_vectorstore(persist_directory, collection_name, embeddings):
    # تابع split() خودتان را اینجا تعریف کنید
    docs = split()  

    # اگر پوشه وجود ندارد، از اول بساز
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
        print(f"📁 پوشه ساخته شد. در حال ایجاد کالکشن جدید...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print(f"✅ کالکشن با {len(docs)} سند ایجاد شد.")
        return vectorstore

    # --------------------------------------------------------
    # پوشه وجود دارد: سعی می‌کنیم به کالکشن متصل شویم
    # --------------------------------------------------------
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    # vectorstore._client.delete_collection("hobbit_text")
    
    # چک کردن تعداد اسناد با استفاده از کلاینت داخلی Chroma
    try:
        # راه اول: استفاده از متد count (مستقیم)
        doc_count = vectorstore._collection.count()
    except Exception:
        # راه دوم (ایمن‌تر): استفاده از متد get
        doc_count = len(vectorstore.get()['ids'])

    # اگر کالکشن خالی است، دوباره پر کن
    if doc_count == 0:
        print(f"⚠️ کالکشن '{collection_name}' خالی است. در حال بازسازی...")
        
        # (اختیاری) پاک کردن کالکشن قبلی برای جلوگیری از تداخل
        try:
            vectorstore._client.delete_collection(collection_name)
        except:
            pass

        # بازسازی از اول
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print(f"✅ کالکشن با {len(docs)} سند بازسازی شد.")
    else:
        print(f"✅ کالکشن با موفقیت بارگذاری شد. تعداد اسناد: {doc_count}")

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
    # -------------------------------------
    # Load Existing DB
    # -------------------------------------
    vectorstore = get_vectorstore(persist_directory, collection_name, embeddings)

    # Access the internal client and delete the collection
    # if os.path.exists(persist_directory):

    #     vectorstore = Chroma(
    #         persist_directory=persist_directory,
    #         embedding_function=embeddings,
    #         collection_name = "hobbit_text"
    #     )
    #     print(f"vector store is loaded {db_name}")

    # else:

    #     docs = split()
    #     print(f"vector store creating \t{db_name}")
    #     print(
    #         f"Total documents: {len(docs)}"
    #     )
    #     vectorstore = Chroma.from_documents(
    #         documents=docs,
    #         embedding=embeddings,
    #         persist_directory=persist_directory,
    #         collection_name = "hobbit_text"
    #     )

    # Assuming you already have a vectorstore object
# vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=your_embeddings)

    # Access the internal Chroma client
    client = vectorstore._client  # Note: this is a private attribute, but it's accessible

    # Now use the same logic as above
    collections = client.list_collections()
    for coll in collections:
        print("vectorstore information")
        print(coll.name, coll.count())
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
    # ret = retriever.invoke("gandalf")
    # print(f"text retriever:\t\t\t\t{ret}")

    return retriever