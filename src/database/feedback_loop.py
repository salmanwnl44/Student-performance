from sqlalchemy.orm import Session
from src.database.models import StudentIntervention

class FeedbackLoop:
    def __init__(self, db_session: Session):
        self.db = db_session

    def record_intervention(self, student_id: str, initial_risk: float, plan: str):
        record = StudentIntervention(
            student_id=student_id,
            initial_risk_score=initial_risk,
            intervention_plan=plan
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record.id

    def update_risk(self, student_id: str, updated_risk: float):
        # Find the latest intervention for this student
        record = self.db.query(StudentIntervention).filter(
            StudentIntervention.student_id == student_id
        ).order_by(StudentIntervention.id.desc()).first()
        
        if record:
            record.updated_risk_score = updated_risk
            if updated_risk < record.initial_risk_score:
                record.success_flag = 1
            else:
                record.success_flag = 0
            self.db.commit()
            return True
        return False

    def get_successful_interventions(self, limit: int = 5):
        records = self.db.query(StudentIntervention).filter(
            StudentIntervention.success_flag == 1
        ).order_by(StudentIntervention.id.desc()).limit(limit).all()
        return [r.intervention_plan for r in records]
