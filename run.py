from dotenv import load_dotenv
load_dotenv()


import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup

import pinecone
from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    GPTVectorStoreIndex,
    QuestionAnswerPrompt,
    PineconeReader
)
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI

# reader = PineconeReader(
#     api_key=os.getenv("PINECONE_API_KEY"),
#     environment="us-west4-gcp"
# )
# docs_from_pinecone = reader.load_data(index_name="nietzsche")

urls = [
    # "https://www.projekt-gutenberg.org/nietzsch/wanderer/wanderer.html",
    # "https://www.projekt-gutenberg.org/nietzsch/wanderer/wande002.html",
    # "https://www.projekt-gutenberg.org/nietzsch/wanderer/wande003.html",
    # "https://www.projekt-gutenberg.org/nietzsch/wanderer/wande004.html",
    "https://policies.google.com/privacy?hl=en-US",
]

def scrape_web(urls):

    for url in urls:
        result = []
        req = requests.get(url)
        soup = BeautifulSoup(req.text, "html.parser")

        # keep only the heading tags up to h3, and p tags
        text = soup.find_all(["a", "h3", "p"])
        # text = soup.find_all(["a"])

        # remove the tags and keep the inner text
        text = [t.text for t in text]

        for i in text:
            try:
                result.append(i.encode('latin').decode("utf-8"))
            except:
                pass

        book_path = Path("web")
        if not book_path.exists():
            book_path.mkdir()

        # pagename = url.split("/")[-1]
        pagename = "data"

        with open(book_path / f"{pagename}.txt", "w") as f:
            f.write("\n".join(result))

def create_pages(urls):

    pages = []
    for url in urls:
        # pagename = url.split("/")[-1]
        pages.append("data")

    return pages

def build_docs(pages):
    docs = {}
    for page in pages:
        docs[page] = SimpleDirectoryReader(
            input_files=[f"web/data.txt"]
        ).load_data()
    return docs

def build_context(model_name):
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name=model_name)
    )
    return ServiceContext.from_defaults(llm_predictor=llm_predictor)

def build_index(pages, docs):

    page_indices = {}
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment="gcp-starter"
    )

    # create a Pinecone index if you don't have one
    # https://openai.com/blog/new-and-improved-embedding-model (12288 -> 1536 dimensions)
    # pinecone.create_index("nietzsche", dimension=1536, metric="cosine")
    
    pinecone_index = pinecone.Index("ailanguage")

    # pinecone_index.upsert("nietzsche_wandere", [1,2,3])
    # pinecone_index.describe_index_stats()
    # pinecone_index.delete_index()

    service_context = build_context("gpt-3.5-turbo")

    for page in pages:
        
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            metadata_filters={"page": page}
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        page_indices[page] = GPTVectorStoreIndex.from_documents(
            docs[page], storage_context=storage_context, service_context=service_context
        )
        page_indices[page].index_struct.index_id = page

    print("Indexing complete.")
    return page_indices

if __name__ == "__main__":
    # scrape_web(urls)
    # assuming books have already been downloaded into your local directory
    pages = create_pages(urls)
    docs = build_docs(pages)
    print(docs.keys())
    indices = build_index(pages, docs)
    response = indices["data"].as_query_engine().query(
        "What kind of user information is collected during the use of your services? Answer in the original English text, and provide an Vietnam translation for the answer"
    )

    print(str(response))


    # PROMPT_TEMPLATE = (
    #     "Here are the context information:"
    #     "\n-----------------------------\n"
    #     "{context_str}"
    #     "\n-----------------------------\n"
    #     "Answer the following question in the original German text, and provide an english translation and explanation in as instructive and educational way as possible: {query_str} \n"
    # )

    # QA_PROMPT = QuestionAnswerPrompt(PROMPT_TEMPLATE)
    # query_engine = indices["wande002.html"].as_query_engine(text_qa_template=QA_PROMPT)
    # response = query_engine.query("What are important things according to Nietzsche?")

    # print(str(response))
    # print(response.get_formatted_sources())