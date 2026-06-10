from ingestion.image_ingestion import build_image_collection
from ingestion.text_ingestion import build_text_collection


def create_db():

    image_retriever = build_image_collection()

    text_retriever = build_text_collection


    return image_retriever, text_retriever
