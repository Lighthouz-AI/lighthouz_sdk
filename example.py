import os

import requests
from lighthouz import Lighthouz
from lighthouz.evaluation import Evaluation


def llamaindex_example_function(query: str) -> str:
    from llama_index import SimpleDirectoryReader, VectorStoreIndex

    DOCUMENTS_FOLDER = "data"
    os.environ["OPENAI_API_KEY"] = "sk-api_key"
    documents = SimpleDirectoryReader(DOCUMENTS_FOLDER).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response


def hf_example_fucntion(query: str) -> str:
    API_URL = (
        "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
    )
    API_KEY = "hf_api_key"
    headers = {f"Authorization": f"Bearer {API_KEY}"}
    payload = {"inputs": query}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]["generated_text"]


def langchain_example_function(query: str) -> str:
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import PyPDFLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.text_splitter import TokenTextSplitter
    from langchain.vectorstores.chroma import Chroma

    os.environ["OPENAI_API_KEY"] = "sk-api_key"
    DOCUMENTS_FOLDER = "data"
    chunk_size = 2000
    chunk_overlap = 150
    embeddings = OpenAIEmbeddings()
    # setting up the vector db
    collection_name = "data-test_vect_embedding"
    local_directory = "data-test_vect_embedding"
    persist_directory = os.path.join(os.getcwd(), local_directory)
    documents = []
    for file in os.listdir(DOCUMENTS_FOLDER):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DOCUMENTS_FOLDER, file)
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
    # main RAG framework
    openai_model = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        temperature=0,
        request_timeout=120,
    )
    retriever = vectDB.as_retriever(return_source_document=True)
    rag_model = RetrievalQA.from_chain_type(
        llm=openai_model,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    response = rag_model({"query": query})["result"]
    return response


lh = Lighthouz("lighthouz-api-key")

evaluation = Evaluation(lh)
# Apps can be created by visiting https://lighthouz.ai/dashboard
e_single = evaluation.evaluate_rag_model(
    response_function=langchain_example_function,
    benchmark_id="benchmark_id",
    app_id="app_id",
)
print(e_single)

e_multiple = evaluation.evaluate_multiple_rag_models(
    response_functions=[langchain_example_function, llamaindex_example_function],
    benchmark_id="benchmark_id",
    app_ids=["app_id1", "app_id2"],
)
print(e_multiple)
