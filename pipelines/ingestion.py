from ingestion.image_ingestion import build_image_collection
from ingestion.text_ingestion import build_text_collection


def create_db(llm):

    image_retriever = build_image_collection(llm)

    text_retriever = build_text_collection()


    return image_retriever, text_retriever
