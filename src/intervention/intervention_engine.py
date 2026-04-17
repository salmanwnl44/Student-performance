from typing import Dict, Any, List
from src.services.llm_service import LLMService
from src.intervention.rules import evaluate_rules

class InterventionEngine:
    def __init__(self):
        self.llm = LLMService()

    def generate_intervention(self, student_data: Dict[str, Any], reasoning: str) -> str:
        flags = evaluate_rules(student_data)
        
        system_prompt = (
            "You are an academic advisor AI. You must read the student's rule-based flags and risk reasoning "
            "and output exactly 3 bullet points outlining a concrete, personalized intervention plan. "
            "Keep the actions practical. For example: 'Meet with academic advisor', 'Follow 7-day study plan', etc."
        )
        
        flags_text = "\n- ".join(flags) if flags else "No specific critical flags triggered."
        
        prompt = (
            f"Rule-based Flags:\n- {flags_text}\n\n"
            f"Reasoning Context:\n{reasoning}\n\n"
            "Provide the 3 Recommended Actions."
        )
        
        return self.llm.generate_response(prompt, system_prompt)
