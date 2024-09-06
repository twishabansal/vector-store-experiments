from langchain_milvus import Milvus
from langchain_google_vertexai import VertexAIEmbeddings
import uuid

from langchain_core.documents import Document

from langchain_milvus import Milvus
from pymilvus import MilvusClient


# The easiest way is to use Milvus Lite where everything is stored in a local file.
# If you have a Milvus server you can use the server URI such as "http://localhost:19530".
URI = "./milvus_example_collections.db"
COLLECTION_NAME = "test_collection"

embeddings = VertexAIEmbeddings(
    model_name="textembedding-gecko@003", project="twisha-dev"
)

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    collection_name=COLLECTION_NAME,
)

# document_1 = Document(
#     page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
#     metadata={"source": "tweet"},
# )

# document_2 = Document(
#     page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
#     metadata={"source": "news"},
# )

# document_3 = Document(
#     page_content="Building an exciting new project with LangChain - come check it out!",
#     metadata={"source": "tweet"},
# )

# document_4 = Document(
#     page_content="Robbers broke into the city bank and stole $1 million in cash.",
#     metadata={"source": "news"},
# )

# document_5 = Document(
#     page_content="Wow! That was an amazing movie. I can't wait to see it again.",
#     metadata={"source": "tweet"},
# )

# document_6 = Document(
#     page_content="Is the new iPhone worth the price? Read this review to find out.",
#     metadata={"source": "website"},
# )

# document_7 = Document(
#     page_content="The top 10 soccer players in the world right now.",
#     metadata={"source": "website"},
# )

# document_8 = Document(
#     page_content="LangGraph is the best framework for building stateful, agentic applications!",
#     metadata={"source": "tweet"},
# )

# document_9 = Document(
#     page_content="The stock market is down 500 points today due to fears of a recession.",
#     metadata={"source": "news"},
# )

# document_10 = Document(
#     page_content="I have a bad feeling I am going to get deleted :(",
#     metadata={"source": "tweet"},
# )

# documents = [
#     document_1,
#     document_2,
#     document_3,
#     document_4,
#     document_5,
#     document_6,
#     document_7,
#     document_8,
#     document_9,
#     document_10,
# ]
# uuids = [str(uuid4()) for _ in range(len(documents))]

# vector_store.add_documents(documents=documents, ids=uuids)

# results = vector_store.similarity_search(
#     "LangChain provides abstractions to make working with LLMs easy",
#     k=2,
#     filter={"source": "tweet"},
# )
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")

# vector_store_loaded = Milvus(
#     embeddings,
#     connection_args={"uri": URI},
#     collection_name="langchain_example",
# )
# for doc in vector_store_loaded.load():
#     print(doc)
# print(vector_store_loaded)

# TODO: Can we get all records without any filter?
# Empty queries don't work

# Filter Works
# all_ids = vector_store.get_pks(expr="source in ['news', 'tweet']")

# Filter Works
# all_ids = vector_store.get_pks(expr="source like '%weather%'")
# print(all_ids)

client = MilvusClient(
    uri=URI
)

# res contains pk, source, text, vector
# Cannot use empty filter
res = client.query(
    collection_name=COLLECTION_NAME,
    filter='pk >= "0"', # Assuming this. All uuids fit into this. 
)
print(res[0].keys())

# Gets all docs by ids
# contains pk, source (metadata here), text, vector
# res = client.get(
#     collection_name=COLLECTION_NAME,
#     ids=["05553c9a-0ed9-4c56-a655-86a103f3526d"]
# )
# print(res)