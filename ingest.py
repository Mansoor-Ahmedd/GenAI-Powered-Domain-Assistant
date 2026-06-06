import os
import chromadb
from chromadb.utils import embedding_functions

# Use ChromaDB's built-in sentence transformer (handles the interface correctly)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu"
)

# Connect to ChromaDB (will create new folder)
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection (delete if exists)
try:
    client.delete_collection("company_docs")
except:
    pass

collection = client.create_collection(
    name="company_docs",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# Load documents from company_docs
doc_folder = "./company_docs"
chunks = []
ids = []

if not os.path.exists(doc_folder):
    print(f"❌ Folder '{doc_folder}' not found. Create it and add .txt files.")
    exit()

for filename in os.listdir(doc_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(doc_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            # Simple chunking (500 words per chunk)
            words = text.split()
            chunk_size = 500
            for j in range(0, len(words), chunk_size):
                chunk = " ".join(words[j:j+chunk_size])
                chunks.append(chunk)
                ids.append(f"{filename}_{j}")
                # Optional: print first few characters to verify
                print(f"Loaded chunk from {filename}: {chunk[:50]}...")

if chunks:
    # Add to ChromaDB
    collection.add(documents=chunks, ids=ids)
    print(f"\n✅ Success! Added {len(chunks)} chunks to ChromaDB")
    print(f"Total documents in collection: {collection.count()}")
else:
    print("❌ No .txt files found in company_docs folder")