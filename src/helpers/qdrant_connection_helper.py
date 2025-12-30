import uuid
import asyncio
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from qdrant_client import models, QdrantClient
from typing import Literal, List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from fastembed.text import TextEmbedding
from fastembed.sparse import SparseTextEmbedding
from fastembed.late_interaction import LateInteractionTextEmbedding

from src.utils.config import settings
from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.text_embedding_helper import embedding_function, text_embedding_model, late_interaction_text_embedding_model, bm25_embedding_model


TEXT_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
LATE_INTERACTION_TEXT_EMBEDDING_MODEL="colbert-ir/colbertv2.0"
BM25_EMBEDDING_MODEL="Qdrant/bm25"


# =============================================================================
# THREAD POOL FOR QDRANT OPERATIONS - Prevents blocking async event loop
# =============================================================================

# Dedicated thread pool for Qdrant sync operations
_qdrant_executor = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="qdrant_sync_"
)

# Timeout for Qdrant operations in seconds
QDRANT_OPERATION_TIMEOUT = 30


# =============================================================================
# SINGLETON QDRANT CLIENT - Prevents multiple connection instances
# =============================================================================

_qdrant_client_instance: Optional[QdrantClient] = None
_qdrant_client_lock = asyncio.Lock()


def get_shared_qdrant_client() -> QdrantClient:
    """
    Get or create shared QdrantClient singleton.
    This prevents creating multiple connections that consume memory.
    """
    global _qdrant_client_instance

    if _qdrant_client_instance is None:
        _qdrant_client_instance = QdrantClient(
            url=settings.QDRANT_ENDPOINT,
            timeout=60,  # Reduced from 600s
            prefer_grpc=True,
        )
        # Log only once
        import logging
        logging.getLogger(__name__).info(
            f"[QDRANT] Created singleton client: {settings.QDRANT_ENDPOINT}"
        )

    return _qdrant_client_instance


class QdrantConnection(LoggerMixin):
    """
    Qdrant connection helper with optimized settings.

    Note: Now uses SINGLETON client to prevent memory issues.
    Timeout reduced from 600s to 60s to fail fast instead of blocking.
    """

    # Reduced timeout to prevent blocking - was 600s
    QDRANT_TIMEOUT = 60

    def __init__(self, embedding_func: HuggingFaceEmbeddings | None = embedding_function):
        super().__init__()
        # Use singleton client instead of creating new one each time
        self.client = get_shared_qdrant_client()
        self.embedding_function = embedding_func

        # These are already cached via @lru_cache in text_embedding_helper
        self.text_embedding_model = text_embedding_model
        self.late_interaction_text_embedding_model = late_interaction_text_embedding_model
        self.bm25_embedding_model = bm25_embedding_model

    # =========================================================================
    # ASYNC EXECUTOR WRAPPER - Run sync Qdrant operations in thread pool
    # =========================================================================

    async def _run_sync_in_executor(
        self,
        func,
        *args,
        timeout: float = QDRANT_OPERATION_TIMEOUT,
        **kwargs
    ):
        """
        Run a synchronous Qdrant operation in thread pool to avoid blocking event loop.

        Args:
            func: The sync function to run
            *args: Positional arguments
            timeout: Operation timeout in seconds
            **kwargs: Keyword arguments

        Returns:
            Result of the function call
        """
        loop = asyncio.get_event_loop()
        try:
            if kwargs:
                func_with_kwargs = partial(func, *args, **kwargs)
                result = await asyncio.wait_for(
                    loop.run_in_executor(_qdrant_executor, func_with_kwargs),
                    timeout=timeout
                )
            else:
                if args:
                    func_with_args = partial(func, *args)
                    result = await asyncio.wait_for(
                        loop.run_in_executor(_qdrant_executor, func_with_args),
                        timeout=timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(_qdrant_executor, func),
                        timeout=timeout
                    )
            return result
        except asyncio.TimeoutError:
            self.logger.warning(f"[QDRANT] Operation timed out after {timeout}s: {func.__name__ if hasattr(func, '__name__') else 'unknown'}")
            raise
        except Exception as e:
            self.logger.error(f"[QDRANT] Error in executor: {e}")
            raise

    # =========================================================================
    # ASYNC WRAPPER METHODS - Non-blocking versions of common operations
    # =========================================================================

    async def collection_exists_async(self, collection_name: str) -> bool:
        """Non-blocking check if collection exists"""
        try:
            return await self._run_sync_in_executor(
                self.client.collection_exists,
                collection_name=collection_name
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"[QDRANT] collection_exists timed out for {collection_name}")
            return False
        except Exception as e:
            self.logger.error(f"[QDRANT] Error checking collection exists: {e}")
            return False

    async def get_collection_async(self, collection_name: str):
        """Non-blocking get collection info"""
        return await self._run_sync_in_executor(
            self.client.get_collection,
            collection_name=collection_name
        )

    async def query_points_async(self, collection_name: str, **kwargs):
        """Non-blocking query points"""
        return await self._run_sync_in_executor(
            self.client.query_points,
            collection_name,
            **kwargs
        )

    async def scroll_async(self, collection_name: str, **kwargs):
        """Non-blocking scroll through collection"""
        return await self._run_sync_in_executor(
            self.client.scroll,
            collection_name=collection_name,
            **kwargs
        )

    async def delete_async(self, collection_name: str, **kwargs):
        """Non-blocking delete operation"""
        return await self._run_sync_in_executor(
            self.client.delete,
            collection_name=collection_name,
            **kwargs
        )

    async def create_payload_index_async(self, collection_name: str, field_name: str, field_schema: str):
        """Non-blocking create payload index"""
        return await self._run_sync_in_executor(
            self.client.create_payload_index,
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema
        )

    async def upload_points_async(self, collection_name: str, points: list, batch_size: int = 16):
        """Non-blocking upload points"""
        return await self._run_sync_in_executor(
            self.client.upload_points,
            collection_name,
            points=points,
            batch_size=batch_size
        )

    async def _create_collection_async(self, collection_name: str) -> bool:
        """Non-blocking create collection"""
        config = self._get_collection_config(
            text_embedding_model=TEXT_EMBEDDING_MODEL,
            late_interaction_text_embedding_model=LATE_INTERACTION_TEXT_EMBEDDING_MODEL,
            bm25_embedding_model=BM25_EMBEDDING_MODEL
        )
        return await self._run_sync_in_executor(
            self.client.create_collection,
            collection_name=collection_name,
            **config
        )

    async def add_data(self,
        documents: List[Document],
        collection_name: str = settings.QDRANT_COLLECTION_NAME,
        organization_id: Optional[str] = None
    ) -> bool:

        # Use async wrapper to avoid blocking event loop
        if not await self.collection_exists_async(collection_name):
            self.logger.info(f"CREATING NEW COLLECTION {collection_name}")
            is_created = await self._create_collection_async(collection_name=collection_name)
            if is_created:
                self.logger.info(f"CREATING NEW COLLECTION {collection_name} SUCCESS.")

        # Upload documents with organization_id (run in executor)
        await self._upload_documents_async(
            collection_name=collection_name,
            documents=documents,
            batch_size=16,
            organization_id=organization_id
        )

        self.logger.info(f"CREATING PAYLOAD INDEX {collection_name}")
        await self.create_payload_index_async(
            collection_name=collection_name,
            field_name="metadata.index",
            field_schema="integer",
        )

        # Create index for organization_id to support efficient searching
        if organization_id:
            await self.create_payload_index_async(
                collection_name=collection_name,
                field_name="metadata.organization_id",
                field_schema="keyword",
            )

        return True


    async def hybrid_search(self,
        query: str = None,
        collection_name: str = settings.QDRANT_COLLECTION_NAME,
        organization_id: Optional[str] = None
    ) -> Optional[List[Document]]:
        """
        Perform hybrid search in Qdrant

        Args:
            query: Query string
            collection_name: Collection name to search
            organization_id: Organization ID (no longer required, keep for compatibility)

        Returns:
            Optional[List[Document]]: Search results
        """
        # Use async wrapper to avoid blocking event loop
        if not await self.collection_exists_async(collection_name):
            raise Exception(f"Collection {collection_name} does not exist")

        dense_query_vector = next(self.text_embedding_model.query_embed(query))
        sparse_query_vector = next(self.bm25_embedding_model.query_embed(query))
        late_query_vector = next(self.late_interaction_text_embedding_model.query_embed(query))

        prefetch = self._create_prefetch(dense_query_vector, sparse_query_vector, late_query_vector)

        # Use async wrapper for query_points
        results = await self.query_points_async(
            collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,  # Reciprocal Rank Fusion
            ),
            with_payload=True,
            limit=20,
        )

        self.logger.info(f"RRF hybrid search returned {len(results.points)} results")
        for i, point in enumerate(results.points[:5]):
            self.logger.info(f"Result {i+1}: id={point.id}, score={point.score:.4f}")

        return [self._point_to_document(point) for point in results.points]


    async def query_headers(
        self,
        documents: List[Document],
        collection_name: str = settings.QDRANT_COLLECTION_NAME,
        organization_id: Optional[str] = None
    ) -> Optional[List[Document]]:

        processed_documents = {}
        # Use async wrapper to get collection info without blocking
        info_collection = await self.get_collection_async(collection_name=collection_name)
        vectors_count = int(info_collection.points_count)
        self.logger.info(f"[HEADERS] Collection {collection_name} has {vectors_count} total points")

        for idx, doc in enumerate(documents):
            doc_name = doc.metadata.get('document_name', 'Unknown')
            headers = doc.metadata.get('headers', 'Unknown')
            doc_id = doc.metadata.get('document_id', 'Unknown')

            self.logger.info(f"[HEADERS] Processing doc {idx+1}/{len(documents)}: {doc_name}, headers={headers[:1000]}...")

            if doc.metadata['headers'] in processed_documents:
                processed_documents[doc.metadata['headers']]['score'] += 1
                self.logger.info(f"[HEADERS] Duplicate headers found, incremented score for {doc_name}, new score: {processed_documents[doc.metadata['headers']]['score']}, content headers: {processed_documents[doc.metadata['headers']]}")
                continue

            self.logger.info(f"[HEADERS] Creating filter for doc_name={doc_name}, headers={headers[:1000]}...")

            # Filter only by document_name and headers
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key="metadata.document_name", match=models.MatchValue(value=doc.metadata['document_name'])),
                    models.FieldCondition(key="metadata.headers", match=models.MatchValue(value=doc.metadata['headers']))
                ]
            )

            self.logger.info(f"[HEADERS] Querying full content with filter for doc_name={doc_name}")

            # Use async wrapper for query_points without blocking
            results = await self.query_points_async(
                collection_name,
                prefetch=[
                    models.Prefetch(
                        filter=query_filter,
                        limit=vectors_count,
                    ),
                ],
                query=models.OrderByQuery(order_by="metadata.index"),
                limit=vectors_count,
            )


            page_content = ''.join([point.payload['page_content'] for point in results.points])
            metadata = {
                'document_name': doc.metadata['document_name'],
                'headers': doc.metadata['headers'],
                'document_id': doc.metadata['document_id'],
            }

            processed_documents[doc.metadata['headers']] = {
                'document': Document(page_content=page_content, metadata=metadata),
                'score': 1
            }

        # Sort the documents based on the 'score' in descending order
        documents_with_scores = processed_documents.items()
        sorted_documents = sorted(documents_with_scores, key=lambda item: item[1]['score'], reverse=True)
        sorted_documents_list = [item[1]['document'] for item in sorted_documents]

        self.logger.info(f"[HEADERS] Sorting completed, returning {len(sorted_documents_list)} processed documents")
        for idx, doc in enumerate(sorted_documents_list[:3]):
            self.logger.info(f"[HEADERS] Final Top {idx+1}: document_name={doc.metadata.get('document_name')}, "
                         f"headers={doc.metadata.get('headers')[:500]}..., content_length={len(doc.page_content)}")

        return sorted_documents_list

    # async def query_headers( 
    #     self,
    #     documents: List[Document],
    #     collection_name: str = settings.QDRANT_COLLECTION_NAME,
    #     organization_id: Optional[str] = None,
    #     max_top_doc_size: int = 8000,
    #     max_other_doc_size: int = 3000,
    #     max_total_chars: int = 16000
    # ) -> Optional[List[Document]]:
    #     """
    #     Query full content for documents while preserving their original order
    #     and applying smart context building for large documents

    #     Args:
    #         documents: List of documents from retrieval/reranking
    #         collection_name: Collection name to query
    #         organization_id: Optional organization ID filter
    #         max_top_doc_size: Maximum size for top-ranked document
    #         max_other_doc_size: Maximum size for other documents
    #         max_total_chars: Maximum total context size

    #     Returns:
    #         List[Document]: Documents with full content in original order
    #     """

    #     # Dictionary to store processed documents with their original positions
    #     processed_documents = []  # Changed from dict to list to preserve all documents

    #     # Get max point data in qdrant collection 
    #     info_collection = self.client.get_collection(collection_name=collection_name)
    #     vectors_count = int(info_collection.points_count)
    #     self.logger.info(f"[HEADERS] Collection {collection_name} has {vectors_count} total points")

    #     # Process each document to get its full content
    #     for idx, doc in enumerate(documents):
    #         doc_name = doc.metadata.get('document_name', 'Unknown')
    #         headers = doc.metadata.get('headers', 'Unknown')
    #         doc_id = doc.metadata.get('document_id', 'Unknown')

    #         self.logger.info(f"[HEADERS] Processing doc {idx+1}/{len(documents)}: {doc_name}, headers={headers[:1000]}...")

    #         # Filter by document_name and headers
    #         query_filter = models.Filter(
    #             must=[
    #                 models.FieldCondition(key="metadata.document_name", match=models.MatchValue(value=doc_name)),
    #                 models.FieldCondition(key="metadata.headers", match=models.MatchValue(value=headers))
    #             ]
    #         )

    #         self.logger.info(f"[HEADERS] Querying full content with filter for doc_name={doc_name}")

    #         # Get all points for this document
    #         results = self.client.query_points(
    #             collection_name,
    #             prefetch=[
    #                 models.Prefetch(
    #                     filter=query_filter,
    #                     limit=vectors_count,
    #                 ),
    #             ],
    #             query=models.OrderByQuery(order_by="metadata.index"),
    #             limit=vectors_count,
    #         )

    #         # Join all content from the points
    #         page_content = ''.join([point.payload['page_content'] for point in results.points])
            
    #         # Apply smart context building by truncating large documents
    #         original_size = len(page_content)
            
    #         # Determine max size based on document rank
    #         max_size = max_top_doc_size if idx == 0 else max_other_doc_size
            
    #         # Truncate if necessary
    #         if original_size > max_size:
    #             page_content = page_content[:max_size] + f"\n\n[Content truncated from {original_size} to {max_size} characters due to length]"
    #             self.logger.info(f"[HEADERS] Truncated document {idx+1}: {original_size} → {max_size} chars")
            
    #         metadata = {
    #             'document_name': doc_name,
    #             'headers': headers,
    #             'document_id': doc_id,
    #             'original_index': idx,  # Store original position
    #             'original_size': original_size,  # Store original size
    #             'is_truncated': original_size > max_size  # Flag if truncated
    #         }

    #         # Store document with its processed content
    #         processed_documents.append(Document(page_content=page_content, metadata=metadata))

    #     # Apply total context size limit while preserving order
    #     result_documents = []
    #     total_chars = 0
        
    #     # Keep track of how many documents we've included
    #     included_docs = 0
        
    #     for doc in processed_documents:
    #         doc_size = len(doc.page_content)
            
    #         # Check if adding this document would exceed max_total_chars
    #         if total_chars + doc_size > max_total_chars:
    #             # If this is the first document, truncate it further
    #             if included_docs == 0:
    #                 available_chars = max_total_chars - 50
    #                 truncated_content = doc.page_content[:available_chars] + f"\n\n[Content further truncated to fit total context limit of {max_total_chars} characters]"
    #                 doc.page_content = truncated_content
    #                 doc.metadata['is_truncated'] = True
    #                 result_documents.append(doc)
    #                 self.logger.info(f"[HEADERS] First document further truncated to {available_chars} chars to fit context limit")
    #                 included_docs += 1
    #             else:
    #                 # Skip this document as it would exceed the limit
    #                 self.logger.info(f"[HEADERS] Document skipped: Would exceed total context limit of {max_total_chars} chars")
    #                 continue
    #         else:
    #             # Document fits within limit, add it
    #             result_documents.append(doc)
    #             total_chars += doc_size
    #             included_docs += 1

    #     self.logger.info(f"[HEADERS] Processing completed, returning {len(result_documents)} documents with total size {total_chars}/{max_total_chars} chars")
    #     for idx, doc in enumerate(result_documents[:3]):
    #         truncated_info = " (truncated)" if doc.metadata.get('is_truncated', False) else ""
    #         self.logger.info(f"[HEADERS] Final Top {idx+1}: document_name={doc.metadata.get('document_name')}, "
    #                         f"headers={doc.metadata.get('headers')[:100]}..., "
    #                         f"size={len(doc.page_content)}/{doc.metadata.get('original_size', 0)} chars{truncated_info}")

    #     return result_documents
    

    def _create_collection(self, collection_name: str) -> bool:

        config = self._get_collection_config(
            text_embedding_model=TEXT_EMBEDDING_MODEL,
            late_interaction_text_embedding_model=LATE_INTERACTION_TEXT_EMBEDDING_MODEL, 
            bm25_embedding_model=BM25_EMBEDDING_MODEL
        )
        return self.client.create_collection(collection_name=collection_name, **config)
    
    
    def _delete_collection(self, collection_name: str) -> bool:
        return self.client.delete_collection(collection_name=collection_name)
    
        
    def _upload_documents(
        self,
        collection_name: str,
        documents: List[Document],
        batch_size: int = 4,
        organization_id: Optional[str] = None
    ) -> None:
        
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        self.logger.info(f"Generating embeddings for {len(documents)} documents using {device_type}")

        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start:batch_start + batch_size]
            
            # Extract page_content for embedding generation
            texts = [doc.page_content for doc in batch]
            dense_embeddings = list(self.text_embedding_model.passage_embed(texts))
            bm25_embeddings = list(self.bm25_embedding_model.passage_embed(texts))
            late_interaction_embeddings = list(self.late_interaction_text_embedding_model.passage_embed(texts))
            
            # Create points with organization_id in metadata
            points = []
            for i, doc in enumerate(batch):
                # Make sure metadata is a dictionary
                metadata = doc.metadata.copy() if isinstance(doc.metadata, dict) else dict(doc.metadata)
                
                # Add organization_id to metadata if present
                if organization_id:
                    metadata['organization_id'] = organization_id
                
                points.append(
                    models.PointStruct(
                        id = str(uuid.uuid4()),
                        vector={
                            TEXT_EMBEDDING_MODEL: dense_embeddings[i].tolist(),
                            LATE_INTERACTION_TEXT_EMBEDDING_MODEL: late_interaction_embeddings[i].tolist(),
                            BM25_EMBEDDING_MODEL: bm25_embeddings[i].as_object(),
                        },
                        payload={
                            "page_content": doc.page_content,
                            "metadata": metadata
                        }
                    )
                )
            
            self.client.upload_points(
                collection_name,
                points=points,
                batch_size=batch_size,
            )

    async def _upload_documents_async(
        self,
        collection_name: str,
        documents: List[Document],
        batch_size: int = 4,
        organization_id: Optional[str] = None
    ) -> None:
        """Async version of _upload_documents - runs in thread pool to avoid blocking"""
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        self.logger.info(f"Generating embeddings for {len(documents)} documents using {device_type}")

        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start:batch_start + batch_size]

            # Extract page_content for embedding generation
            texts = [doc.page_content for doc in batch]
            dense_embeddings = list(self.text_embedding_model.passage_embed(texts))
            bm25_embeddings = list(self.bm25_embedding_model.passage_embed(texts))
            late_interaction_embeddings = list(self.late_interaction_text_embedding_model.passage_embed(texts))

            # Create points with organization_id in metadata
            points = []
            for i, doc in enumerate(batch):
                # Make sure metadata is a dictionary
                metadata = doc.metadata.copy() if isinstance(doc.metadata, dict) else dict(doc.metadata)

                # Add organization_id to metadata if present
                if organization_id:
                    metadata['organization_id'] = organization_id

                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            TEXT_EMBEDDING_MODEL: dense_embeddings[i].tolist(),
                            LATE_INTERACTION_TEXT_EMBEDDING_MODEL: late_interaction_embeddings[i].tolist(),
                            BM25_EMBEDDING_MODEL: bm25_embeddings[i].as_object(),
                        },
                        payload={
                            "page_content": doc.page_content,
                            "metadata": metadata
                        }
                    )
                )

            # Use async wrapper to upload points without blocking
            await self.upload_points_async(collection_name, points=points, batch_size=batch_size)

    def _point_to_document(self, point: models.ScoredPoint) -> Document:
        return Document(page_content=point.payload['page_content'], metadata=point.payload['metadata'])

    def _create_prefetch(
        self, 
        dense_query_vector,
        sparse_query_vector, 
        late_query_vector,
        query_filter: Optional[models.Filter] = None
    ) -> List[models.Prefetch]:
        return [
            models.Prefetch(
                query=dense_query_vector,
                using=TEXT_EMBEDDING_MODEL,
                filter=query_filter,
                limit=40,   
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_query_vector.as_object()),
                using=BM25_EMBEDDING_MODEL,
                filter=query_filter,
                limit=40,
            ),
            # Late interaction prefetch
            models.Prefetch(
                query=late_query_vector,
                using=LATE_INTERACTION_TEXT_EMBEDDING_MODEL,
                filter=query_filter,
                limit=40,  
            ),
        ]
    
    def _create_headers_filter(self, metadata: dict, organization_id: Optional[str] = None) -> models.Filter:
        conditions = [
            models.FieldCondition(key="metadata.document_name", match=models.MatchValue(value=metadata['document_name'])),
            models.FieldCondition(key="metadata.headers", match=models.MatchValue(value=metadata['headers']))
        ]
        
        # Add filter for organization_id if any
        if organization_id:
            conditions.append(
                models.FieldCondition(key="metadata.organization_id", match=models.MatchValue(value=organization_id))
            )
            
        return models.Filter(must=conditions)
        
    async def delete_document_by_file_name(
            self,
            document_name: str = None,
            collection_name: str = settings.QDRANT_COLLECTION_NAME,
            organization_id: Optional[str] = None
    ):
        try:
            # Create filter with document_name and organization_id (if present)
            conditions = [
                models.FieldCondition(
                    key="metadata.document_name",
                    match=models.MatchValue(value=document_name),
                )
            ]

            if organization_id:
                conditions.append(
                    models.FieldCondition(
                        key="metadata.organization_id",
                        match=models.MatchValue(value=organization_id),
                    )
                )

            # Use async wrapper to avoid blocking event loop
            await self.delete_async(
                collection_name=collection_name,
                points_selector=models.Filter(must=conditions),
            )
        except Exception as e:
            self.logger.error('event=delete-document-by-file-name-in-qdrant '
                                'message="Delete document by file name in Qdrant Failed. '
                                f'error="Got unexpected error." error="{str(e)}"')

    async def delete_document_by_batch_ids(
            self,
            document_ids: list[str] = None,
            collection_name: str = settings.QDRANT_COLLECTION_NAME,
            organization_id: Optional[str] = None
    ):
        try:
            # Create filter for document_ids and organization_id (if any)
            conditions = [
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchValue(value=document_id)
                )
                for document_id in document_ids
            ]

            filter_params = models.Filter(should=conditions)

            # Add organization_id condition if present
            if organization_id:
                filter_params = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.organization_id",
                            match=models.MatchValue(value=organization_id),
                        )
                    ],
                    should=conditions
                )

            # Use async wrapper to avoid blocking event loop
            await self.delete_async(
                collection_name=collection_name,
                points_selector=filter_params,
            )
        except Exception as e:
            self.logger.error('event=delete-document-by-batch-ids-in-qdrant '
                              'message="Delete document by batch ids in Qdrant Failed. '
                              f'error="Got unexpected error." error="{str(e)}"')
        

    def _get_embedding_dim(self, model_name: str, model_type: Literal['text', 'sparse_text', 'late_interaction_text']):
        if model_type == 'text':
            supported_models = TextEmbedding.list_supported_models()
        elif model_type == 'sparse_text':
            supported_models = SparseTextEmbedding.list_supported_models()
        elif model_type == 'late_interaction_text':
            supported_models = LateInteractionTextEmbedding.list_supported_models()  

        for model in supported_models:
            if model['model'] == model_name:
                return model['dim']
        return None

    def _get_collection_config(self,
        text_embedding_model: str,
        late_interaction_text_embedding_model: str,
        bm25_embedding_model: str,
    ) -> Dict[str, Any]:
        
        text_embedding_dim = self._get_embedding_dim(model_name=text_embedding_model, model_type='text')
        late_interaction_text_embedding_dim = self._get_embedding_dim(model_name=late_interaction_text_embedding_model, model_type='late_interaction_text')

        return {
            "vectors_config": {
                text_embedding_model: models.VectorParams(
                    size=text_embedding_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True
                ),
                late_interaction_text_embedding_model: models.VectorParams(
                    size=late_interaction_text_embedding_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    )
                ),
            },
            "sparse_vectors_config": {
                bm25_embedding_model: models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True,
                    ),
                    modifier=models.Modifier.IDF,
                )
            },
            "hnsw_config": models.HnswConfigDiff(
                on_disk=True,
                m=16,  # Số connections
                ef_construct=100,  # Build quality
            ),
            "optimizers_config": models.OptimizersConfigDiff(
                memmap_threshold=10000,
                indexing_threshold=10000,
            ),
            "quantization_config": models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=False,
                ),
            ),
            "on_disk_payload": True,
            "timeout": 600
        }