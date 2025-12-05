
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class NewsItem(BaseModel):
    title: str
    publisher_name: str
    published_date: datetime
    url: str
    symbols: Optional[List[str]] = None
    text: Optional[str] = None 