from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings    
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter

from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint



#NORMAL MODEL

MODEL = "llama2"

model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

parser = StrOutputParser()

template = """
Answer the question based on the context below. If you can't
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")


def without_RAG(questions):
    answer_without_RAG = model.invoke(questions)
    print("sending response")
    return answer_without_RAG


def with_RAG(questions):
    #RAG PROCESSING

    loader = PyPDFLoader("the_resource.pdf") ###RAG PDF
    pages = loader.load_and_split()
    pages


    vectorstore_RAG = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    retriever_RAG = vectorstore_RAG.as_retriever()



    chain_RAG = (
        {
            "context": itemgetter("question") | retriever_RAG,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | parser
    )

    return chain_RAG.invoke({'question': questions})


def with_JSON(questions):
    #JSON 

    loader = JSONLoader(file_path="the_database.json", jq_schema=".", text_content=False)

    documents = loader.load()
    vectorstore_JSON = DocArrayInMemorySearch.from_documents(documents, embedding=embeddings)

    retriever_JSON = vectorstore_JSON.as_retriever()

    from operator import itemgetter

    chain_JSON = (
        {
            "context": itemgetter("question") | retriever_JSON,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | parser
    )

    return chain_JSON.invoke({'question': questions})