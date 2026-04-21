"""
RAG context provider — no ChromaDB, no vector DB.
Returns curated institutional advice based on department + risk level.
"""

# Curated teacher notes keyed by (department, risk_level)
# Fallback keys: ("*", risk_level) and ("*", "*")
_CONTEXT_BANK: dict[tuple[str, str], str] = {
    ("Computer Science", "High"): (
        "Students in CS with high dropout risk often struggle with foundational programming courses. "
        "Pairing them with a senior peer mentor for weekly debug sessions has shown strong results. "
        "Recommend breaking large projects into daily checkpoints and using office hours proactively."
    ),
    ("Computer Science", "Medium"): (
        "CS students at medium risk benefit from structured study groups focused on algorithms and data structures. "
        "Encourage participation in hackathons to rebuild motivation and practical confidence."
    ),
    ("Engineering", "High"): (
        "High-risk Engineering students frequently fall behind on lab reports and group projects. "
        "Tutoring support for mathematics and physics fundamentals is the most impactful intervention. "
        "Consider reduced credit load for students working part-time."
    ),
    ("Engineering", "Medium"): (
        "Engineering students at medium risk often need better time management frameworks. "
        "Weekly planner reviews and deadline reminders from advisors reduce missed submissions significantly."
    ),
    ("Business", "High"): (
        "High-risk Business students respond well to career counselling sessions that reconnect them with their goals. "
        "Group case study workshops and peer accountability partners have improved retention."
    ),
    ("Business", "Medium"): (
        "Business students benefit from networking events and internship placement support to stay motivated. "
        "Faculty check-ins at the mid-semester point help catch early disengagement."
    ),
    ("Medicine", "High"): (
        "Medicine students with high dropout risk often report burnout and exam anxiety. "
        "Mental health counselling, study skills workshops, and realistic goal-setting are critical interventions."
    ),
    ("Medicine", "Medium"): (
        "Medical students at medium risk benefit from exam simulation sessions and anatomy revision workshops. "
        "Peer-led study circles have shown measurable improvement in retention."
    ),
    ("Arts", "High"): (
        "High-risk Arts students often lack a clear career pathway, driving disengagement. "
        "Portfolio review sessions with faculty and industry mentors help re-establish purpose and direction."
    ),
    ("Arts", "Medium"): (
        "Arts students at medium risk respond well to collaborative project showcases and creative community events. "
        "Flexible assignment deadlines and regular feedback loops from instructors reduce stress."
    ),
    ("Science", "High"): (
        "High-risk Science students frequently struggle with laboratory performance and report writing. "
        "Targeted lab assistant sessions and structured report templates reduce failure rates."
    ),
    ("Science", "Medium"): (
        "Science students at medium risk benefit from additional problem-solving workshops and past exam reviews. "
        "Group study sessions supervised by teaching assistants improve engagement."
    ),
    # Generic fallbacks
    ("*", "High"): (
        "Students with high dropout risk need immediate, personalised outreach from their academic advisor. "
        "A concrete action plan covering attendance, study hours, and academic support referrals is essential. "
        "Early intervention within the first two weeks of a declining trend is most effective."
    ),
    ("*", "Medium"): (
        "Students at medium risk benefit from proactive check-ins and goal-setting sessions. "
        "Connecting them with campus support services (tutoring, counselling, career guidance) preventatively "
        "can stop risk from escalating to high."
    ),
    ("*", "Low"): (
        "Low-risk students should be encouraged to maintain their current habits and explore enrichment opportunities "
        "such as research projects, leadership roles, or advanced coursework."
    ),
}


class RAGRetriever:
    """Retrieves relevant institutional context for a student based on department and risk level."""

    def get_context_for_student(self, student_department: str, risk_level: str) -> str:
        # Try exact match first, then department wildcard, then full wildcard
        for dept_key in (student_department, "*"):
            key = (dept_key, risk_level)
            if key in _CONTEXT_BANK:
                return _CONTEXT_BANK[key]
        return "No specific historical notes available."
