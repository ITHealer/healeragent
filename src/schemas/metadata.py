import uuid
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

# Schema for RAG

class Metadata(BaseModel):
    document_id: str = Field(description='The document ID', default=None)
    document_name: str = Field(description='Document name', default=None)
    page: int = Field(description='The page number of page', default=None)
    is_parent: bool = Field(description='Indicate documnet is all page or not', default=False)


class Page(BaseModel):
    content: str = Field(description='The raw content in page relation with object', default='')
    metadata: Metadata = Field(description='The metadata of page')


class MetadataAnswer(BaseModel):
    document_id: str
    pages: list[int]


class KeywordPromptFormat(BaseModel):
    keywords: str = Field(description='All keywords has extracted in user query by LLM')


class DocSource(BaseModel):
    document_id: str = Field(description='Document ID of source in Document Context')
    pages: List[str] = Field(description='The list of unique page numbers with document name')


class ObjectAnswer(BaseModel):
    answer: str = Field(description='The answer has been provided by LLM')
    sources: List[DocSource] = Field(description='All sources have content related to the answer')


class AnswerResponse(BaseModel):
    answer: str = Field(description='The answer has been provided by LLM')
    sources: List[Metadata] = Field(description='The metadata of answer')


class SuggestQuestionsResponse(BaseModel):
    questions: List[str] = Field(description='List string suggested questions')


class Document(BaseModel):
    file_name: str = Field(description='The name of the file')
    extension: str = Field(description='The MIME type of the file')
    product: Optional[str] = Field(description='The related product', default=None)
    root_url: Optional[str] = Field(description='The root URL of the page', default=None)
    minio_url: Optional[str] = Field(description='The public URL on Minio', default=None)
    page_url: Optional[str] = Field(description='The page URL', default=None)
    size: Optional[float] = Field(description='The size of the file', default=None)
    mutable: bool = Field(description='Whether the file can be changed', default=True)
    created_at: datetime = Field(description='The creation time', default_factory=datetime.utcnow)
    updated_at: datetime = Field(description='The last update time', default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(description='Create by current user', default=None)