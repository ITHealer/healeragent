from src.database.base_connection import MySQLConnectionBase


class MySQLFrontendConnection(MySQLConnectionBase):
    """MySQL connection for frontend database"""
    
    def __init__(self):
        super().__init__(db_config_key='MYSQL')