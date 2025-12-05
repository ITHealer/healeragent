from src.database.base_connection import PostgreSQLConnectionBase


class PostgreSQLConnection(PostgreSQLConnectionBase):
    """PostgreSQL database connection"""
    
    def __init__(self):
        super().__init__()