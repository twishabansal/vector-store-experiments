import chromadb


chroma_client = chromadb.PersistentClient(path="db")

# chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="my_collection3")
collection2 = chroma_client.create_collection(name="my_collection2")

collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

collection2.add(
    documents=[
        "This is a document about phones",
        "This is a document about laptops"
    ],
    ids=["id12", "id22"]
)

results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)