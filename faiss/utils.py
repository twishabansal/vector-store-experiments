import pickle

path = "faiss_index/index.pkl"
with open(path, "rb") as f:
    content = pickle.load(f)

print(content)