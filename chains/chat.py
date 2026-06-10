from ingestion.ingestion import build_db

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


def split_questions(questions):
  """Split questions into sub-questions"""
  temp = []
  for q in questions:
    if len(q) > 0 and q[0] in ["1", "2", "3", "4"]:
      # print(q)
      temp.append(q)

  return temp

def decomposition(question: str, llm):
    # Decomposition

    logs = []

    logs.append(
        "[✓] Generating sub queries"
    )
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
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

    ("system", """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise."""),
        ("human", """Question: {question}

    Context: {context}

    Answer:""")

    ])
    sub_questions, logs = decomposition(question, llm)
    rag_results = []
    docs = []
    for sub_question in sub_questions:
      retrieved_docs = retriever.invoke(sub_question)
      # docs.append(retrieved_docs)
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

    # response = final_rag_chain.invoke({"context": context, "question": question})
    return result, context, {"logs": logs, "retrieved_chunks": docs}



def get_answers(llm, question):
   
   image_retriever, text_retriever = build_db()

   text_result, context, metadata = generate_answer(question, llm, text_retriever)

   img_results = image_retriever.invoke(
    question
    )
   

   return {"text_answer": text_result,
           "img_result": img_results.metadata["image_path"]}
