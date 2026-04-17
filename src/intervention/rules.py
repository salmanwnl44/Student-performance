from typing import Dict, Any, List

def evaluate_rules(student_data: Dict[str, Any]) -> List[str]:
    """Evaluates simple threshold rules for interventions."""
    flags = []
    
    attendance = student_data.get("attendance_rate", 100.0)
    gpa = student_data.get("gpa_current", 4.0)
    fees_pending = student_data.get("scholarship_status", "None") == "None" and student_data.get("family_income", "High") == "Low"
    missed_deadlines = student_data.get("missed_deadlines_count", 0)
    
    if attendance < 75.0:
        flags.append("Attendance is critically low.")
    if gpa < 2.5:
        flags.append("Academic performance (GPA) is concerningly low.")
    if missed_deadlines > 3:
        flags.append("Student is missing multiple deadlines consecutively.")
    if fees_pending:
        flags.append("Student may be facing financial constraints.")
        
    return flags
