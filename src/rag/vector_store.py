import os
import chromadb

class VectorStore:
    def __init__(self, db_path: str = "./chroma_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name="student_context")

    def add_documents(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query: str, n_results: int = 3):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
