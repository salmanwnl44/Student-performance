from src.rag.vector_store import VectorStore

class RAGRetriever:
    def __init__(self):
        self.vector_store = VectorStore()

    def get_context_for_student(self, student_department: str, risk_level: str) -> str:
        query = f"teacher notes and historical interventions for {risk_level} risk students in {student_department}"
        results = self.vector_store.search(query, n_results=2)
        
        if not results['documents'] or not results['documents'][0]:
            return "No specific historical notes available."
            
        context = "\n".join(results['documents'][0])
        return context
