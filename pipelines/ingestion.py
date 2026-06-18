from pipelines.image_ingestion import build_image_collection
from pipelines.text_ingestion import build_text_collection


def create_db(llm):

    image_retriever, image_vectorstore = build_image_collection(llm)

    text_retriever = build_text_collection()


    return image_vectorstore, text_retriever
