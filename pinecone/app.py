import getpass
import os
import time

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_alloydb_pg import (
    AlloyDBEngine,
    Column,
    AlloyDBVectorStore,
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from uuid import uuid4

from langchain_core.documents import Document

PROJECT_ID = os.getenv("PROJECT_ID")
INSTANCE_NAME = os.getenv("INSTANCE_NAME")
REGION = os.getenv("REGION")
CLUSTER = os.getenv("CLUSTER")
DATABASE = os.getenv("DATABASE")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "test-index"  # change if desired

EMBEDDINGS_SERVICE = VertexAIEmbeddings(
    model_name="textembedding-gecko@003", project=PROJECT_ID
)
ENGINE = AlloyDBEngine.from_instance(
    project_id=PROJECT_ID,
    instance=INSTANCE_NAME,
    region=REGION,
    cluster=CLUSTER,
    database=DATABASE,
    user=USER,
    password=PASSWORD,
)


def create_pinecone_index():
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

    return pc.Index(INDEX_NAME)


def get_pinecone_index():
    return pc.Index(INDEX_NAME)


def populate_pinecone_index(index):
    embeddings = VertexAIEmbeddings(
        model_name="textembedding-gecko@003", project=PROJECT_ID
    )
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    document_1 = Document(
        page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet", "location": "l1"},
    )

    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news", "location": "l2"},
    )

    document_3 = Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet", "location": "l2"},
    )

    document_4 = Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news", "location": "l1"},
    )

    document_5 = Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "tweet", "location": "l1"},
    )

    document_6 = Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website", "location": "l2"},
    )

    document_7 = Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website", "location": "l1"},
    )

    document_8 = Document(
        page_content="LangGraph is the best framework for building stateful, agentic applications!",
        metadata={"source": "tweet", "location": "l2"},
    )

    document_9 = Document(
        page_content="The stock market is down 500 points today due to fears of a recession.",
        metadata={"source": "news", "location": "l3"},
    )

    document_10 = Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet", "location": "l3"},
    )

    documents = [
        document_1,
        document_2,
        document_3,
        document_4,
        document_5,
        document_6,
        document_7,
        document_8,
        document_9,
        document_10,
    ]
    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)

    results = vector_store.similarity_search(
        "LangChain provides abstractions to make working with LLMs easy",
        k=2,
        filter={"source": "tweet"},
    )
    print("Num results found: ", len(results))
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")


# Source: https://docs.pinecone.io/guides/data/list-record-ids#paginate-through-results
def get_all_pinecone_data(index):
    results = index.list_paginated(prefix="")
    ids = [v.id for v in results.vectors]
    while results.pagination is not None:
        pagination_token = results.pagination.next
        results = index.list_paginated(prefix="", pagination_token=pagination_token)
        ids.extend([v.id for v in results.vectors])
    # Contains namespace, usage and vectors
    # Vectors contain uuid, id, metadata and tex
    all_data = index.fetch(ids)
    ids = []
    embeddings = []
    content = []
    metadatas = []
    for doc in all_data["vectors"].values():
        ids.append(doc["id"])
        embeddings.append(doc["values"])
        content.append(doc["metadata"]["text"])
        metadata = doc["metadata"]
        del metadata["text"]
        metadatas.append(metadata)
    return ids, embeddings, content, metadatas


def migrate_pinecone(pinecone_index):
    ENGINE.init_vectorstore_table(
        table_name=INDEX_NAME,
        vector_size=768,
        metadata_columns=[Column("source", "VARCHAR"), Column("location", "VARCHAR")],
    )
    vector_store = AlloyDBVectorStore.create_sync(
        engine=ENGINE,
        embedding_service=EMBEDDINGS_SERVICE,
        table_name=INDEX_NAME,
        metadata_columns=["source", "location"],
    )
    ids, embeddings, content, metadatas = get_all_pinecone_data(pinecone_index)
    vector_store.add_embeddings(
        texts=content,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


if __name__ == "__main__":
    # index = create_pinecone_index()
    # populate_pinecone_index(index)
    # data = get_all_pinecone_data(index)
    index = get_pinecone_index()
    migrate_pinecone(index)
