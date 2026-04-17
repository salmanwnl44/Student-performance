from typing import Dict, Any
from src.services.llm_service import LLMService

class ReasoningEngine:
    def __init__(self):
        self.llm = LLMService()

    def generate_reasoning(self, student_data: Dict[str, Any], risk_probability: float) -> str:
        risk_level = "High" if risk_probability > 0.6 else "Medium" if risk_probability > 0.3 else "Low"
        risk_percentage = round(risk_probability * 100, 2)
        
        system_prompt = (
            "You are an AI Student Retention Assistant. Your job is to analyze the following student data "
            "and explain why the student has the given dropout risk in plain, supportive English. "
            "Do not output raw numbers unless they are important (like attendance or GPA). Keep the response under 60 words."
        )
        
        student_attributes = "\n".join([f"{k}: {v}" for k, v in student_data.items() if k not in ["student_id", "final_grade"]])
        
        prompt = (
            f"Student Data:\n{student_attributes}\n\n"
            f"Dropout Risk Probability: {risk_percentage}% ({risk_level} Risk)\n\n"
            "Generate a brief explanation."
        )
        
        return self.llm.generate_response(prompt, system_prompt)
