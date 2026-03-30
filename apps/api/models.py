from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Paper(Base):
    __tablename__ = 'papers'
    
    paper_id = Column(String, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    keywords = Column(Text)
    authors = Column(Text)
    year = Column(Integer)
    doi = Column(String)
    journal = Column(String)
    url = Column(String)
    source_path = Column(String)
    ingest_status = Column(String) # ok, missing_abstract, needs_ocr, error
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ScreeningLabel(Base):
    __tablename__ = 'screening_labels'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(String, ForeignKey('papers.paper_id'), index=True)
    ai_label = Column(String) # include, exclude, uncertain
    ai_score = Column(Float)
    threshold_used = Column(Float)
    rationale_json = Column(JSON)
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class ReviewAction(Base):
    __tablename__ = 'review_actions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(String, ForeignKey('papers.paper_id'), index=True)
    reviewer_id = Column(String)
    action = Column(String) # override, confirm, flag
    final_label = Column(String)
    comment = Column(Text)
    prev_label = Column(String)
    prev_score = Column(Float)
    model_version = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Cluster(Base):
    __tablename__ = 'clusters'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(String, ForeignKey('papers.paper_id'), index=True)
    cluster_id = Column(String, index=True)
    dup_group_id = Column(String)
    cluster_repr_paper_id = Column(String)
    method = Column(String) # hdbscan, kmeans, none
    created_at = Column(DateTime, default=datetime.utcnow)
