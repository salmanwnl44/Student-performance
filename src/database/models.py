from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class StudentIntervention(Base):
    __tablename__ = 'interventions'
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, index=True)
    initial_risk_score = Column(Float)
    intervention_plan = Column(String)
    updated_risk_score = Column(Float, nullable=True)
    success_flag = Column(Integer, default=0) # 1 if updated_risk_score < initial_risk_score

engine = create_engine('sqlite:///student_feedback.db', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
