from pipelines.ingestion import create_db

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


def split_questions(questions):
  """Split questions into sub-questions"""
  temp = []
  for q in questions:
    if len(q) > 0 :
      print(f"generated queries:\t\t\t{q}")
      temp.append(q)

  return temp

def decomposition(question: str, llm):
    # Decomposition

    logs = []

    logs.append(
        "[✓] Generating sub queries"
    )
    template = """
    You are an expert retrieval assistant for a knowledge base built from J.R.R. Tolkien's The Hobbit book.

    Your task is to generate multiple search queries that improve retrieval quality from the text collection.

    The collection contains the original text of The Hobbit book.

    When generating queries:

    - Identify important characters, locations, objects, and events.
    - Include alternative phrasings and synonyms when appropriate.
    - Expand references and pronouns into explicit entity names whenever possible.
    - Consider related terms, aliases, and book-specific terminology.
    - Preserve the original meaning of the user's question.
    - Generate diverse queries that explore different textual aspects of the question.

    User question:
    {question}

    Return exactly 3 search queries, one per line, without numbering, explanations, or additional text.
    """
    prompt_decomposition = ChatPromptTemplate.from_template(template)


    generate_queries_decomposition = (prompt_decomposition
                                      | llm
                                      | StrOutputParser()
                                      | (lambda x: x.split("\n"))
                                      | RunnableLambda(split_questions))
    sub_questions = generate_queries_decomposition.invoke({"question": question})
    return sub_questions, logs


def retrieve_and_rag(question,  sub_question_generator_chain, retriever, llm):
    """Rag on each sub-question"""

    prompt = ChatPromptTemplate.from_messages([

    ("system", """
    You are an expert question-answering assistant for J.R.R. Tolkien's The Hobbit book.

    Use only the provided context to answer the user's question.

    Instructions:

    - Base your answer strictly on the retrieved context.
    - Do not use outside knowledge.
    - If the answer cannot be determined from the context, say "I don't know based on the provided context."
    - Keep the answer concise and accurate.
    - When possible, mention relevant characters, locations, or events explicitly.
    """),

        ("human", """
    Question:
    {question}

    Context:
    {context}

    Answer:
    """)
    ])

    sub_questions, logs = decomposition(question, llm)
    rag_results = []
    docs = []
    for sub_question in sub_questions:
      retrieved_docs = retriever.invoke(sub_question)
      for doc in retrieved_docs:

            docs.append(
                {
                    "query": question,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get(
                        "score",
                        None
                    ),
                }
            )
      answer = (prompt | llm | StrOutputParser()).invoke({"context": retrieved_docs,
                                                          "question": sub_question})
    rag_results.append(answer)
    logs.append(
        f"[✓] Retrieved {len(docs)} unique chunks"
    )

    return rag_results, sub_questions, logs, docs




def format_qa_pairs(answers, questions):
  """Format Q and A pairs"""

  formatted_string = ""
  for i, (question, answer) in enumerate(zip(questions, answers), start=1):
    formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
  return formatted_string.strip()


def generate_answer(question, llm, retriever):
    answers, questions, logs, docs = retrieve_and_rag(question, decomposition(question, llm), retriever, llm)
    context = format_qa_pairs(answers, questions)
    docs = docs[0:2]
    # Prompt
    template = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    logs.append(
        f"[✓] Generated final answer"
    )
    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    result = final_rag_chain.invoke({"context": context, "question": question})

    return result, context, {"logs": logs, "retrieved_chunks": docs}



def get_answers(llm, question, image_vectorstore, text_retriever):
   

    text_result, context, metadata = generate_answer(question, llm, text_retriever)

    img_results = image_vectorstore.similarity_search_with_score(
    question,
    k=4
    )

    THRESHOLD = 1.44

    img_result = []

    for doc, score in img_results:
        print(f"retrieved image score:\t\t{score}")
        print(f"retrieved image content:\t\t{doc}")
        print("------------------------")

        if score < THRESHOLD:
            img_result.append(doc)
            print(f"retrieved image score:\t\t{score}")
    
    MAX_IMAGES = 3

    img_result = img_result[:MAX_IMAGES]

    return {"text_answer": text_result,
           "img_result": img_result}
