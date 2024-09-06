This is a POC for getting bulk data from various vector stores: 
* ChromaDB
* Milvus
* Qdrant
* Pinecone
* Weaviate

FAISS is not yet implememnted.
I opted for the native clients of various vector stores (for data retrieval) as their Langchain interfaces lacked methods to fetch all documents.