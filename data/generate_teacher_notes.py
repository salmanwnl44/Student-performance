import os
import sys

# Add root folder to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.rag.vector_store import VectorStore

def generate_mock_notes():
    store = VectorStore()
    
    docs = [
        "High risk students in Computer Science usually struggle because they miss programming lab deadlines. Setting up a paired programming session helped student X last term.",
        "Engineering students with low attendance often face commute issues. Offering recorded lectures improved retention by 20%.",
        "Business department students with pending fees are 40% more likely to drop out. Connecting them to the fast-track scholarship office is recommended.",
        "Arts students with low GPA responded well to early portfolio reviews.",
        "High risk Medical students showed improvement when assigned to an older student mentor."
    ]
    
    metadatas = [
        {"department": "Computer Science", "risk_context": "High"},
        {"department": "Engineering", "risk_context": "High"},
        {"department": "Business", "risk_context": "High"},
        {"department": "Arts", "risk_context": "Medium"},
        {"department": "Medicine", "risk_context": "High"},
    ]
    
    ids = [f"note_{i}" for i in range(len(docs))]
    
    store.add_documents(docs, metadatas, ids)
    print("Mock teacher notes added to ChromaDB.")

if __name__ == "__main__":
    generate_mock_notes()
