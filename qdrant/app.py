import asyncio
import os
from uuid import uuid4

from langchain_core.documents import Document
from langchain_google_alloydb_pg import (AlloyDBEngine, AlloyDBVectorStore,
                                         Column)
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

PROJECT_ID = os.getenv("PROJECT_ID")
INSTANCE_NAME = os.getenv("INSTANCE_NAME")
REGION = os.getenv("REGION")
CLUSTER = os.getenv("CLUSTER")
DATABASE = os.getenv("DATABASE")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")

EMBEDDINGS_SERVICE = VertexAIEmbeddings(
    model_name="textembedding-gecko@003", project=PROJECT_ID
)

CLIENT = QdrantClient(path="/tmp/langchain_qdrant_test")

COLLECTION_NAME = "test_qdrant"

def create_qdrant_vectorstore():
    CLIENT.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=CLIENT,
        collection_name=COLLECTION_NAME,
        embedding=EMBEDDINGS_SERVICE,
    )

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
        "LangChain provides abstractions to make working with LLMs easy", k=2
    )
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")


def get_all_data():
    docs = CLIENT.scroll(collection_name=COLLECTION_NAME, with_vectors=True)
    id = []
    content = []
    vector = []
    metadatas = []
    for doc in docs[0]:
        id.append(doc.id)
        content.append(doc.payload["page_content"])
        vector.append(doc.vector)
        metadatas.append(doc.payload["metadata"])
    return id, content, vector, metadatas


async def migrate_qdrant():
    engine = await AlloyDBEngine.afrom_instance(
        project_id=PROJECT_ID,
        instance=INSTANCE_NAME,
        region=REGION,
        cluster=CLUSTER,
        database=DATABASE,
        user=USER,
        password=PASSWORD,
    )
    await engine.ainit_vectorstore_table(
        table_name=COLLECTION_NAME,
        vector_size=768,
        metadata_columns=[Column("source", "VARCHAR"), Column("location", "VARCHAR")],
    )
    vector_store = await AlloyDBVectorStore.create(
        engine=engine,
        embedding_service=EMBEDDINGS_SERVICE,
        table_name=COLLECTION_NAME,
        metadata_columns=["source", "location"],
    )
    ids, content, embeddings, metadatas = get_all_data()
    await vector_store.aadd_embeddings(
        texts=content,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


if __name__ == "__main__":
    # create_qdrant_vectorstore()
    # get_all_data()
    asyncio.run(migrate_qdrant())
