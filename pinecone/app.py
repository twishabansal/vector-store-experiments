import getpass
import os
import time
import numpy as np
import asyncio

from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from uuid import uuid4

from langchain_core.documents import Document

PROJECT_ID = os.getenv("PROJECT_ID")

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
index_name = "test-index"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

embeddings = VertexAIEmbeddings(
    model_name="textembedding-gecko@003", project=PROJECT_ID
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

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
# print("Num results found: ", len(results))
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")


# Source: https://stackoverflow.com/questions/75894927/pinecone-can-i-get-all-dataall-vector-from-a-pinecone-index-to-move-data-i
def get_ids_from_query(index,input_vector):
  print("searching pinecone...")
  results = index.query(
    top_k=10000,
    include_values=False,
    include_metadata=False,
    vector=input_vector,
  )
  ids = set()
  for result in results['matches']:

    ids.add(result.id)
  return ids

def get_all_ids_from_index(index, num_dimensions, namespace=""):
  num_vectors = index.describe_index_stats()
  num_vectors = num_vectors.namespaces[namespace].vector_count
  all_ids = set()
  while len(all_ids) < num_vectors:
    print("Length of ids list is shorter than the number of total vectors...")
    input_vector = np.random.rand(num_dimensions).tolist()
    print("creating random vector...")
    ids = get_ids_from_query(index,input_vector)
    print("getting ids from a vector query...")
    all_ids.update(ids)
    print("updating ids set...")
    print(f"Collected {len(all_ids)} ids out of {num_vectors}.")
  return all_ids


async def print_all_docs():
    all_ids = get_all_ids_from_index(index, num_dimensions=768)
    all_docs = await vector_store.aget_by_ids(all_ids)
    print("Num all docs:", len(all_docs))
    for doc in all_docs:
        print(doc.page_content)

if __name__ == "__main__":
    asyncio.run(print_all_docs())
