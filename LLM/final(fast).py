from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings    
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter

import spacy
from typing import AsyncIterator, Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

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


#RAG PROCESSING

loader = PyPDFLoader("iitg_dataset.pdf") ###RAG PDF
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


#NER PROCESSING
   
nlp = spacy.load("/home/raone/LLM Project/NER/model-best") ###NER MODEL  

all_data = []

for i in range(len(pages)):
    doc = nlp(pages[i].page_content)
    for ent in doc.ents:
        data = " is ".join([ent.text, ent.label_])  # Join the elements with " is "
        all_data.append([data])  # Append the formatted data as a list to all_data

all_data




class CustomDocumentLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            for line in f:
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path},
                )
                line_number += 1


with open("./NER.txt", "w", encoding="utf-8") as f:
    f.write(str(all_data))

CustomDocumentLoader = CustomDocumentLoader("./NER.txt")
CustomDocumentLoader.load()

pages += CustomDocumentLoader.load()
pages


vectorstore_NER = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever_NER = vectorstore_NER.as_retriever()

chain_NER = (
    {
        "context": itemgetter("question") | retriever_NER,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
)


def without_RAG(questions):
    answer_without_RAG = model.invoke(questions)
    print("sending without_RAG response")
    return answer_without_RAG


def with_RAG(questions):
    print("sending with_RAG response")
    return chain_RAG.invoke({'question': questions})


def with_NER(questions):
    print("sending with_NER response")
    return chain_NER.invoke({'question': questions})

 
