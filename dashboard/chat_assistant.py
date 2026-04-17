import streamlit as st
import requests

def render_chat_assistant():
    st.header("💬 AI Student Mentor")
    st.markdown("Ask the AI mentor questions about your performance and get actionable advice.")
    
    with st.expander("📝 Set Up Your Profile Options", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            gpa = st.number_input("Current GPA", min_value=0.0, max_value=4.5, value=2.8)
            attendance = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=78.0)
            department = st.selectbox("Department", ["Computer Science", "Engineering", "Business", "Medicine", "Arts", "Science"])
        with col2:
            missed_deadlines = st.number_input("Missed Deadlines", min_value=0, max_value=20, value=2)
            login_freq = st.number_input("LMS Login Frequency", min_value=0, max_value=50, value=8)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask how you can improve..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Formulate payload
        student_data = {
            "gpa_current": gpa,
            "attendance_rate": attendance,
            "department": department,
            "missed_deadlines_count": missed_deadlines,
            "login_frequency": login_freq,
            # Defaults for API validation
            "age": 20, "gender": "Male", "semester": 3, "parent_education": "Bachelor",
            "family_income": "Middle", "commute_distance": 10.0, "scholarship_status": 0,
            "has_internet_access": 1, "part_time_job": 0, "midterm_score": 65.0, 
            "assignment_score": 70.0, "quiz_score": 60.0, "forum_participation_score": 3,
            "time_spent_on_materials": 10.0, "study_hours_weekly": 8.0, "library_usage_weekly": 2
        }
        
        payload = {
            "student": student_data,
            "message": prompt
        }

        with st.chat_message("assistant"):
            with st.spinner("Analyzing profile & history..."):
                try:
                    response = requests.post("http://localhost:8000/chat/student", json=payload)
                    response.raise_for_status()
                    answer = response.json().get("response", "No answer found.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error communicating with AI: {e}")
