## Imports
import os

import requests

from lighthouz import Lighthouz
from lighthouz.app import App
from lighthouz.benchmark import Benchmark
from lighthouz.evaluation import Evaluation

## SET VARIABLES
os.environ[
    "OPENAI_API_KEY"
] = "sk-xx"  # Enter your OpenAI key to be used in the RAG model
# RAG_DOCUMENT = "DATA_FOLDER/FILENAME_TO_GENERATE_RAG_BENCHMARK" # Enter the file name where RAG data is present, example EXAMPLE-DATA/apple-10Q-Q2-2022.pdf
RAG_DOCUMENT = "./EXAMPLE-DATA/apple-10K-2022.pdf"  # this example file is present at https://github.com/Lighthouz-AI/lighthouz_sdk/blob/88a6887398ed646909bf1ca085e40be94a5938bc/EXAMPLE-DATA/apple-10Q-Q2-2022.pdf

lh = Lighthouz(
    "LIGHTHOUZ-API-KEY"
)  # To obtain a Lighthouz API key contact srijan@lighthouz.ai
benchmark_category = [
    "rag_benchmark"
]  # list of benchmarks to be created, options are ["rag_benchmark", "out_of_context", "pii_leak", "prompt_injection"]

## STEP 1: Generate a RAG benchmark with Lighthouz AutoBench
### You can two options: 1. use a benchmark you created earlier, or 2. create a new benchmar

### Option 1. You can provide benchmark ids of benchmarks you created previously. For example, we have pre-loaded a finance benchmark in your account. This benchmark has financial queries generated from apple's 10-K 2022 report.
### Benchmark id is available on the lighthouz dashboard.
benchmark_id = "659b66198e4cc1f4af4e2373"  # this is the pre-loaded finance benchmark on apple's 10-K report.

### Option 2. You can generate a new benchmark by providing it a document or folder with documents. AutoBench will generate benchmarks based on the information in the document(s).
### Uncomment the following code to generate new benchmarks
# benchmark_generator = Benchmark(lh)
# benchmark_data = benchmark_generator.generate_benchmark(
#     file_path=RAG_DOCUMENT, benchmark_category=benchmark_category
# )
# benchmark_id = benchmark_data["benchmark_id"]

## STEP 2: Register your RAG app on Lighthouz
### STEP 2a: Create your RAG app. This is an example of a RAG app built using LangChain, OpenAI, and ChomaDB. You can replace it with your own RAG app code.
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import HuggingFaceEndpoint, HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )

    # LLM used in RAG
    if llm == "gpt-3.5-turbo":
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
        input_variables=["context", "question"], template=prompt_template
    )

    rag_model = RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    print("Langchain RAG OpenAI agent has been initialized.")
    return rag_model


rag_model = langchain_rag_model(llm="gpt-3.5-turbo")


### The following function allows Lighthouz to send queries to your RAG app.
### If you replace the RAG app code above, you will need to update this function that takes the query prompt string as input and returns a textual response.
def langchain_rag_query_function(query: str) -> str:
    """
    This is a function to send queries to the RAG model
    """
    response = rag_model({"query": query})["result"]
    return response


### Step 2b: Register your app

#### You have two options: 1) register a new app, or 2) use an app you created earlier
#### Option 1: You can register a new app.

### Note: Only register an application once to track all its evals one place. After first registration, use its app_id
app = App(lh)
app_data = app.register(name="gpt-3.5-turbo", model="gpt-3.5-turbo")
app_id = app_data["app_id"]

#### Option 2: Use the id of an existing app.
# If you want to use pre-registered app, comment the above 3 lines that register the app and add your app id below.
# app_id = "659d3a7f2d63d34f8fe49ca1"

## Step 3: Evaluate the RAG app on the benchmark with Lighthouz AutoEval
evaluation = Evaluation(lh)
e_single = evaluation.evaluate_rag_model(
    response_function=langchain_rag_query_function,
    benchmark_id=benchmark_id,
    app_id=app_id,
)

## Step 4: Compare multiple RAG apps on the benchmark with Lighthouz Arena

#### First, I create another RAG app, using the same RAG code defined above, but using a different LLM (GPT-4).
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

#### The following code compares two RAG apps on the same benchmark.
e_multiple = evaluation.evaluate_multiple_rag_models(
    response_functions=[
        langchain_rag_query_function,
        langchain_rag_query_function_gpt4,
    ],
    benchmark_id=benchmark_id,
    app_ids=[app_id, app_id_gpt4],
)
print(e_multiple)
