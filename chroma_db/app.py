import os
from uuid import uuid4
from langchain_google_alloydb_pg import AlloyDBEngine, Column, AlloyDBVectorStore, AlloyDBLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

PROJECT_ID = os.getenv("PROJECT_ID")
INSTANCE_NAME = os.getenv("INSTANCE_NAME")
REGION = os.getenv("REGION")
CLUSTER = os.getenv("CLUSTER")
DATABASE = os.getenv("DATABASE")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")

COLLECTION_NAME = "test_chroma"
EMBEDDINGS_SERVICE = VertexAIEmbeddings(
    model_name="textembedding-gecko@003", project=PROJECT_ID
)
VECTOR_STORE = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=EMBEDDINGS_SERVICE,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
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


def create_chroma_collection():
    document_1 = Document(
        page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet", "location": "test"},
        id=1,
    )

    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news", "location": "test2"},
        id=2,
    )

    document_3 = Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet", "location": "test1"},
        id=3,
    )

    document_4 = Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news", "location": "test2"},
        id=4,
    )

    document_5 = Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "tweet", "location": "test"},
        id=5,
    )

    document_6 = Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website", "location": "test"},
        id=6,
    )

    document_7 = Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website", "location": "test3"},
        id=7,
    )

    document_8 = Document(
        page_content="LangGraph is the best framework for building stateful, agentic applications!",
        metadata={"source": "tweet", "location": "test1"},
        id=8,
    )

    document_9 = Document(
        page_content="The stock market is down 500 points today due to fears of a recession.",
        metadata={"source": "news", "location": "test2"},
        id=9,
    )

    document_10 = Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet", "location": "test"},
        id=10,
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

    VECTOR_STORE.add_documents(documents=documents, ids=uuids)
    # results = VECTOR_STORE.similarity_search("Will it be hot tomorrow?", k=1)
    # print(results)


def get_chroma_docs():
    docs = VECTOR_STORE.get(include=["metadatas", "documents", "embeddings"])
    # for doc in docs:
    #     print(f"{doc}: {docs[doc]}")
    return docs


def migrate_chroma():
    ENGINE.init_vectorstore_table(
        table_name=COLLECTION_NAME,
        vector_size=768,
        metadata_columns=[Column("source", "VARCHAR"), Column("location", "VARCHAR")],
    )
    vector_store = AlloyDBVectorStore.create_sync(
        engine=ENGINE,
        embedding_service=EMBEDDINGS_SERVICE,
        table_name=COLLECTION_NAME,
        metadata_columns=["source", "location"],
    )
    docs = get_chroma_docs()
    # print("curr doc metadatas", docs["metadatas"])
    vector_store.add_embeddings(
        texts=docs["documents"],
        embeddings=docs["embeddings"],
        metadatas=docs["metadatas"],
        ids=docs["ids"],
    )


def check_collection_in_alloy():
    table_name = COLLECTION_NAME
    loader = AlloyDBLoader.create_sync(
        engine=ENGINE,
        query=f"SELECT * FROM {table_name};",
    )
    all_docs_num = 0
    for doc in loader.load():
        # print(doc)
        all_docs_num += 1
    print("Num docs: ", all_docs_num)


# Run the main function
if __name__ == "__main__":
    # create_chroma_collection()
    # get_chroma_docs()
    migrate_chroma()
    # check_collection_in_alloy()
