
import os

import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader, TextLoader


#from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader

questions = [
    "What did the Paul Graham do growing up?",
    "What did the Paul Graham during his school days?",
    "What languages did Paul Graham use?",
    "Who was Rich Draves?",
    "What happened to the Paul Graham in the summer of 2016?",
    "What happened to the Paul Graham in the fall of 1992?",
    #"How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?",
    #"How much was allocated to a implement a means-tested dental care program in the 2023 Canadian federal budget?",
]

prompt = PromptTemplate.from_template("""
You are a helpful assistant that can answer questions based on the context provided and not prior knowledge.

Context:
{context}

Question:
{question}

Answer:
""")


def test_indexing():

    llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    #from langchain_openai import OpenAIEmbeddings

    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_store = Chroma(embedding_function=embeddings)

    #canada_loader = PyPDFLoader("data/2023_canadian_budget.pdf")
    #canada_docs = canada_loader.load()

    loader = TextLoader("data/input/paul_graham_essay.txt")
    paul_graham_docs = loader.load()

    docs = paul_graham_docs # + canada_docs

    text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=50)
    all_splits = text_splitter.split_documents(docs)

    print(f"Number of chunks: {len(all_splits)}")

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering


    def rag_query(question: str, verbose : bool = False):
        retrieved_docs = vector_store.similarity_search(question)

        if(verbose):
            for doc in retrieved_docs:
                print(f"Page: {doc.page_content} {doc.metadata}")

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        messages = prompt.invoke({"question": question, "context": docs_content})
        response = llm.invoke(messages)
        return response.content


    for question in questions:
        print(f"Question: {question} *************************************************************\n")
        print(f"Answer: {rag_query(question, verbose=False)}")
        print("\n\n")


def rag_query(question: str, vector_store: Chroma, llm: BaseChatModel, verbose : bool = False):
    retrieved_docs = vector_store.similarity_search(question)

    if(verbose):
        for doc in retrieved_docs:
            print(f"Page: {doc.page_content} {doc.metadata}")

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    messages = prompt.invoke({"question": question, "context": docs_content})
    response = llm.invoke(messages)
    return response.content


async def index_folder_contents(folder_path: str, verbose: bool = False):

    # Load all text, PDF, or other supported files
    loader = DirectoryLoader(folder_path, glob="**/*.*", show_progress=True)

    # Load documents into LangChain format
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(documents)

    # Index chunks
    db_persist_directory = "data/vector"
    embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"
    indexed_files_directory = "data/indexed"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vector_store = Chroma(embedding_function=embeddings, persist_directory=db_persist_directory)

    #add documents to vector store
    vector_store.add_documents(documents=all_splits)

    #create 'indexed' folder if it doesn't exist
    if not os.path.exists(indexed_files_directory):
        os.makedirs(indexed_files_directory)

    if not os.path.exists(f"{indexed_files_directory}/{embeddings_model_name}"):
        os.makedirs(f"{indexed_files_directory}/{embeddings_model_name}")

    #Move indexed files from input folder to 'indexed' folder
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), "r") as f:
            #shutil.move(os.path.join(folder_path, file), os.path.join(indexed_files_directory, file))
            pass

    return vector_store

#store = asyncio.run(index_folder_contents("data/input"))

print("Application started")
test_indexing()



