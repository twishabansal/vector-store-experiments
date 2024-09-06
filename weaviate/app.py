import weaviate
import weaviate.classes as wvc
# from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
import os

# Set these environment variables
URL = os.getenv("WEAVIATE_URL")
API_KEY = os.getenv("WCS_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
EMBEDDINGS_API_KEY = os.getenv("COHERE_API_KEY")
  
# # Connect to a WCS instance
# client = weaviate.connect_to_wcs(
#     cluster_url=URL,
#     auth_credentials=weaviate.auth.AuthApiKey(API_KEY))

# client = weaviate.Client(
#     url = URL,
#     auth_client_secret=weaviate.auth.AuthApiKey(api_key=API_KEY),
# )

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=URL,
    auth_credentials=weaviate.auth.AuthApiKey(API_KEY),
    headers={"X-Cohere-Api-Key": EMBEDDINGS_API_KEY},
)

embeddings = VertexAIEmbeddings(
    model_name="textembedding-gecko@003", project=PROJECT_ID
)

try:
    test_collection = client.collections.create(
        name="test_collection2",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_cohere(),  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
        generative_config=wvc.config.Configure.Generative.cohere()  # Ensure the `generative-openai` module is used for generative queries
    )
    for i in range(10):
        test_collection.data.insert(properties={"page_content": f"data_{i}", "metadata": {"col_0":f"val_0_{i}", "col_1":f"val_1_{i}"}})
    
    collection = client.collections.get("test_collection2")
    for item in collection.iterator():
        print(item.uuid, item.properties)
        # Example: 24a4c33e-f66d-400c-897c-ed75d5f2c7cc {'metadata': {'col_1': 'val_1_4', 'col_0': 'val_0_4'}, 'page_content': 'data_4'}

finally:
    client.close()  # Close client gracefully