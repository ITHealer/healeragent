from src.database.base_connection import MySQLConnectionBase


class MySQLNewsConnection(MySQLConnectionBase):
    """MySQL connection for news database"""
    
    def __init__(self):
        super().__init__(db_config_key='MYSQL_CRAWLER')