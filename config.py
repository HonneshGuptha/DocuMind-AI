"""
DocuMind-AI Configuration Module
Defines application settings and paths for SQLite database, upload storage,
and file validation.
"""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    """Base application configuration."""
    
    BASE_DIR = BASE_DIR
    
    # Secret key for session management and CSRF protection
    SECRET_KEY = os.environ.get('SECRET_KEY', 'documind-ai-dev-secret-key-2026')
    
    # Storage paths
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'documind.db')
    
    # File upload restrictions
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB maximum file size
    
    # Application settings
    DEBUG = True
    TESTING = False


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True


class TestingConfig(Config):
    """Testing environment configuration."""
    TESTING = True
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'test_documind.db')
