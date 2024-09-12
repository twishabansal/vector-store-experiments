import asyncio
import os

from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore, Column

import weaviate
import weaviate.classes as wvc

# Set these environment variables
URL = os.getenv("WEAVIATE_URL")
API_KEY = os.getenv("WCS_API_KEY")
EMBEDDINGS_API_KEY = os.getenv("COHERE_API_KEY")

PROJECT_ID = os.getenv("PROJECT_ID")
INSTANCE_NAME = os.getenv("INSTANCE_NAME")
REGION = os.getenv("REGION")
CLUSTER = os.getenv("CLUSTER")
DATABASE = os.getenv("DATABASE")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")

COLLECTION_NAME = "test_weaviate_collection"


def create_weaviate_vectorstore():
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
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=URL,
            auth_credentials=weaviate.auth.AuthApiKey(API_KEY),
            headers={"X-Cohere-Api-Key": EMBEDDINGS_API_KEY},
        )
        test_collection = client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_cohere(),
            generative_config=wvc.config.Configure.Generative.cohere(),
        )
        for doc in documents:
            test_collection.data.insert(
                properties={"page_content": doc.page_content, "metadata": doc.metadata}
            )
    finally:
        client.close()


def get_all_data():
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=URL,
            auth_credentials=weaviate.auth.AuthApiKey(API_KEY),
            headers={"X-Cohere-Api-Key": EMBEDDINGS_API_KEY},
        )
        ids = []
        content = []
        embeddings = []
        metadatas = []
        collection = client.collections.get(COLLECTION_NAME)
        for item in collection.iterator(include_vector=True):
            ids.append(str(item.uuid))
            content.append(item.properties["page_content"])
            embeddings.append(item.vector["default"])
            metadatas.append(item.properties["metadata"])
    finally:
        client.close()
    return ids, content, embeddings, metadatas


async def migrate_weaviate():
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=URL,
            auth_credentials=weaviate.auth.AuthApiKey(API_KEY),
            headers={"X-Cohere-Api-Key": EMBEDDINGS_API_KEY},
        )
        EMBEDDINGS_SERVICE = CohereEmbeddings(
            async_client=client,
            model="embed-english-v3.0",
            cohere_api_key=EMBEDDINGS_API_KEY,
        )
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
            vector_size=1024,
            metadata_columns=[
                Column("source", "VARCHAR"),
                Column("location", "VARCHAR"),
            ],
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
    finally:
        client.close()


if __name__ == "__main__":
    # create_weaviate_vectorstore()
    # ids, content, embeddings, metadatas = get_all_data()
    asyncio.run(migrate_weaviate())
