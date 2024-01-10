import os
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.llms import HuggingFaceHub, HuggingFaceEndpoint
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from lighthouz import Lighthouz
from lighthouz.app import App
from lighthouz.benchmark import Benchmark
from lighthouz.evaluation import Evaluation


## SET VARIABLES
os.environ["OPENAI_API_KEY"] = "" # Enter your OpenAI key to be used in the RAG model
RAG_DOCUMENT = "DATA_FOLDER/FILENAME_TO_GENERATE_RAG_BENCHMARK" # Enter the file name where RAG data is present, example EXAMPLE-DATA/apple-10Q-Q2-2022.pdf
# RAG_DOCUMENT = "EXAMPLE-DATA/apple-10Q-Q2-2022.pdf" 

lh = Lighthouz("LH-API-KEY") # To obtain a Lighthouz API key contact srijan@lighthouz.ai
benchmark_category = ["rag_benchmark", "out_of_context", "pii_leak", "prompt_injection"] # list of benchmarks to be created 

def langchain_rag_model(llm="gpt-3.5-turbo"):
    """
    This is a RAG model built with Langchain, OpenAI, and Chroma
    """
    print("Initializing LangChain RAG OpenAI Agent")

    chunk_size = 2000
    chunk_overlap = 150
    collection_name = "data-test_vect_embedding"
    local_directory = "data-test_vect_embedding"
    persist_directory = os.path.join(os.getcwd(), local_directory)
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        embeddings = OpenAIEmbeddings()
        documents = []
        if RAG_DOCUMENT.endswith(".pdf"): 
            loader = PyPDFLoader(RAG_DOCUMENT)
            documents.extend(loader.load())
        else:
            for file in os.listdir(RAG_DOCUMENT):
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(RAG_DOCUMENT, file)
                    loader = PyPDFLoader(pdf_path)
                    documents.extend(loader.load())
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splitdocument = text_splitter.split_documents(documents)
        vectDB = Chroma.from_documents(
            splitdocument,
            embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        vectDB.persist()
    else:
        # Load the existing vector store
        embeddings = OpenAIEmbeddings()
        vectDB = Chroma(
            collection_name=collection_name, persist_directory=persist_directory, embedding_function=embeddings
        )

    # LLM used in RAG
    if llm =="gpt-3.5-turbo":
        llm_model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        request_timeout=120,
        )
    elif llm == "gpt-4": 
        llm_model = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            request_timeout=120,
            )

    retriever = vectDB.as_retriever(return_source_document=True)

    # prepare stuff prompt template
    prompt_template = """You are a helpful assistant. Your job is to provide the answer for the question based on the given context. 
    ## CONTEXT: {context}
    ## QUESTION: {question}
    ## ANSWER: """.strip()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    rag_model = RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt":prompt}
    )
    print("Langchain RAG OpenAI agent has been initialized.")
    return rag_model


def langchain_rag_query_function(query: str) -> str:
    """
    This is a function to send queries to the RAG model 
    """
    response = rag_model({"query": query})["result"]
    return response

rag_model = langchain_rag_model(llm="gpt-3.5-turbo")


## Register application. 
### Note: Only register an application once to track all its evals one place. After first registration, use its app_id
app = App(lh)
app_data = app.register(name="gpt-3.5-turbo", model="gpt-3.5-turbo")
app_id = app_data["app_id"]

## Generate benchmark
benchmark_generator = Benchmark(lh)
benchmark_data = benchmark_generator.generate_benchmark(file_path=RAG_DOCUMENT, benchmark_category=benchmark_category) 
benchmark_id = benchmark_data["benchmark_id"]

## Run evaluations on a benchmark
evaluation = Evaluation(lh)
e_single = evaluation.evaluate_rag_model(
    response_function=langchain_rag_query_function,
    benchmark_id=benchmark_id, # You can also enter an existing benchmark id. Benchmark ids are on the dashboard: https://lighthouz.ai/benchmarks/
    app_id=app_id, 
)
print(e_single)


## Compare multiple models on a benchmark 
### First creating another RAG model using gpt-4
rag_model_gpt4 = langchain_rag_model(llm="gpt-4")

def langchain_rag_query_function_gpt4(query: str) -> str:
    """
    This is a function to ask queries to the RAG model with GPT4
    """
    response = rag_model_gpt4({"query": query})["result"]
    return response

app = App(lh)
app_data = app.register(name="gpt-4", model="gpt-4")
app_id_gpt4 = app_data["app_id"]

e_multiple = evaluation.evaluate_multiple_rag_models(
    response_functions=[langchain_rag_query_function, langchain_rag_query_function_gpt4],
    benchmark_id=benchmark_id,
    app_ids=[app_id, app_id_gpt4],
)
print(e_multiple)
