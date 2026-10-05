"""Neural Embeddings & Vector Database Store Module.

Features:
- EmbeddingEngine: Unified embedding interface supporting:
    1. OpenAI text-embedding-3-small (with API authentication & fallback)
    2. PubMedBERT (pritamdeka/S-PubMedBert-MS-MARCO)
    3. BioBERT (dmis-lab/biobert-v1.1)
    4. SentenceTransformers all-MiniLM-L6-v2
- VectorStoreManager:
    - ChromaDB (Persistent vector store with metadata filtering)
    - FAISS (In-memory IndexFlatIP cosine similarity search with disk serialization)
- Unified search results with score normalization and parent-chunk expansion.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from config import EmbeddingModelType, VectorDBType, settings
from src.chunking import TextChunk

logger = logging.getLogger(__name__)

# Suppress HuggingFace hub symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VectorSearchResult:
    """Standardized search hit returned across ChromaDB and FAISS."""
    chunk_id: str
    text: str
    score: float  # Cosine similarity (0.0 to 1.0; higher is more relevant)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_name: str = ""
    page_number: int = 1
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "source_name": self.source_name,
            "page_number": self.page_number,
            "parent_id": self.parent_id,
            "parent_text": self.parent_text,
        }


# =============================================================================
# EMBEDDING ENGINE
# =============================================================================

class EmbeddingEngine:
    """Generates transformer neural embeddings across OpenAI and Biomedical models."""

    # Model name aliases and paths
    MODEL_REGISTRY: Dict[str, str] = {
        "text-embedding-3-small": "text-embedding-3-small",
        "PubMedBERT": "pritamdeka/S-PubMedBert-MS-MARCO",
        "BioBERT": "dmis-lab/biobert-v1.1",
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    }

    _model_cache: Dict[str, Any] = {}

    def __init__(self, model_name: Union[EmbeddingModelType, str] = settings.DEFAULT_EMBEDDING_MODEL) -> None:
        self.raw_model_name = model_name.value if isinstance(model_name, EmbeddingModelType) else model_name
        self.resolved_model_path = self.MODEL_REGISTRY.get(self.raw_model_name, self.raw_model_name)
        self.is_openai = "text-embedding" in self.raw_model_name.lower()

        self.openai_client = None
        self.local_model = None
        self._dim: Optional[int] = None

        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize OpenAI client or HuggingFace local SentenceTransformer."""
        if self.is_openai:
            api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
            if api_key:
                try:
                    from openai import OpenAI
                    self.openai_client = OpenAI(api_key=api_key)
                    self._dim = 1536
                    logger.info("Initialized OpenAI Embedding Client (%s)", self.raw_model_name)
                    return
                except Exception as e:
                    logger.warning("OpenAI client initialization failed: %s. Falling back to MiniLM.", e)

            logger.info("No valid OpenAI API key found; falling back to local sentence-transformers MiniLM.")
            self.is_openai = False
            self.resolved_model_path = "sentence-transformers/all-MiniLM-L6-v2"

        # Local SentenceTransformer
        if self.resolved_model_path in self._model_cache:
            self.local_model = self._model_cache[self.resolved_model_path]
            self._dim = self._get_dim(self.local_model)
        else:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer model: %s", self.resolved_model_path)
                self.local_model = SentenceTransformer(self.resolved_model_path)
                self._dim = self._get_dim(self.local_model)
                self._model_cache[self.resolved_model_path] = self.local_model
                logger.info("Successfully loaded %s (dimension=%d)", self.resolved_model_path, self._dim)
            except Exception as e:
                logger.error("Failed to load %s: %s. Using default all-MiniLM-L6-v2.", self.resolved_model_path, e)
                from sentence_transformers import SentenceTransformer
                fallback_path = "sentence-transformers/all-MiniLM-L6-v2"
                self.local_model = SentenceTransformer(fallback_path)
                self._dim = self._get_dim(self.local_model)
                self._model_cache[fallback_path] = self.local_model

    def _get_dim(self, model: Any) -> int:
        """Helper to get embedding dimension cleanly across sentence-transformers versions."""
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        if self._dim is not None:
            return self._dim
        if self.is_openai:
            return 1536
        if self.local_model:
            return self._get_dim(self.local_model)
        return 384

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate normalized embedding vectors for a list of document strings."""
        if not texts:
            return []

        # 1. OpenAI Embeddings
        if self.is_openai and self.openai_client:
            try:
                all_embeddings: List[List[float]] = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i: i + batch_size]
                    clean_batch = [t.replace("\n", " ") if t else " " for t in batch]
                    res = self.openai_client.embeddings.create(
                        input=clean_batch,
                        model=self.raw_model_name,
                    )
                    batch_embs = [d.embedding for d in res.data]
                    all_embeddings.extend(batch_embs)
                return all_embeddings
            except Exception as e:
                logger.warning("OpenAI embedding API failed: %s. Falling back to local model.", e)

        # 2. Local SentenceTransformer Embeddings
        clean_texts = [t.strip() if t and t.strip() else "medical empty" for t in texts]
        embeddings = self.local_model.encode(
            clean_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generate normalized embedding vector for a single search query."""
        results = self.embed_documents([text])
        return results[0] if results else [0.0] * self.dimension


# =============================================================================
# BASE VECTOR STORE
# =============================================================================

class BaseVectorStore(ABC):
    """Abstract Base Class for Vector Store implementations."""

    @abstractmethod
    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """Embed and index chunks into vector database."""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search vector database by natural language query."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total indexed vectors."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Purge all vectors in the store."""
        pass


# =============================================================================
# CHROMADB VECTOR STORE
# =============================================================================

class ChromaVectorStore(BaseVectorStore):
    """Persistent ChromaDB vector database manager with metadata filtering."""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        persist_dir: Union[str, Path] = settings.CHROMA_PERSIST_DIR,
        collection_name: str = settings.CHROMA_COLLECTION_NAME,
    ) -> None:
        self.embedding_engine = embedding_engine
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        import chromadb
        self.client = chromadb.PersistentClient(path=str(self.persist_dir.resolve()))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB initialized at '%s' (Collection: '%s')", self.persist_dir, self.collection_name)

    def _sanitize_metadata(self, chunk: TextChunk) -> Dict[str, Union[str, int, float, bool]]:
        """Flatten nested metadata dictionaries into Chroma-compatible primitives."""
        clean_meta: Dict[str, Union[str, int, float, bool]] = {
            "chunk_id": str(chunk.chunk_id),
            "source_name": str(chunk.source_name),
            "source_type": str(chunk.source_type),
            "page_number": int(chunk.page_number),
            "token_count": int(chunk.token_count),
            "strategy": str(chunk.strategy),
            "section_header": str(chunk.section_header or ""),
            "cluster_id": int(chunk.cluster_id if chunk.cluster_id is not None else 0),
            "cluster_label": str(chunk.cluster_label or "General Medicine"),
            "parent_id": str(chunk.parent_id or ""),
        }
        # Encode complex metadata like entities/acronyms as JSON string
        if "entity_summary" in chunk.metadata:
            clean_meta["entity_summary_json"] = json.dumps(chunk.metadata["entity_summary"])
        if "acronyms" in chunk.metadata:
            clean_meta["acronyms_json"] = json.dumps(chunk.metadata["acronyms"])
        if chunk.parent_text:
            clean_meta["has_parent"] = True

        return clean_meta

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """Add and embed chunks into ChromaDB."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedding_engine.embed_documents(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [self._sanitize_metadata(c) for c in chunks]

        # Chroma upsert in batches of 100
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            self.collection.upsert(
                ids=ids[i: i + batch_size],
                embeddings=embeddings[i: i + batch_size],
                documents=texts[i: i + batch_size],
                metadatas=metadatas[i: i + batch_size],
            )
        logger.info("Upserted %d chunks into ChromaDB (Total: %d)", len(chunks), self.count())

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Perform dense semantic similarity search in ChromaDB."""
        query_embedding = self.embedding_engine.embed_query(query)

        where_clause = None
        if filter_metadata:
            # Format simple equality filters for Chroma
            filter_items = [{k: {"$eq": v}} if not isinstance(v, dict) else {k: v} for k, v in filter_metadata.items()]
            where_clause = {"$and": filter_items} if len(filter_items) > 1 else filter_items[0]

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, max(1, self.count())),
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB query error: %s", e)
            return []

        search_hits: List[VectorSearchResult] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return search_hits

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for cid, doc_text, meta, dist in zip(ids, docs, metas, dists):
            # Chroma returns cosine distance = 1 - cosine_similarity (for cosine space)
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            parent_id = meta.get("parent_id") or None
            search_hits.append(VectorSearchResult(
                chunk_id=cid,
                text=doc_text,
                score=similarity,
                metadata=meta,
                source_name=meta.get("source_name", "Unknown Source"),
                page_number=int(meta.get("page_number", 1)),
                parent_id=parent_id if parent_id else None,
            ))

        return search_hits

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' cleared.", self.collection_name)


# =============================================================================
# FAISS VECTOR STORE
# =============================================================================

class FAISSVectorStore(BaseVectorStore):
    """In-memory and file-persistent FAISS IndexFlatIP (Cosine Similarity) store."""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        index_dir: Union[str, Path] = settings.FAISS_INDEX_DIR,
    ) -> None:
        self.embedding_engine = embedding_engine
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "faiss.index"
        self.metadata_file = self.index_dir / "faiss_metadata.json"

        import faiss
        self.faiss = faiss
        self.dimension = self.embedding_engine.dimension
        self.index = self.faiss.IndexFlatIP(self.dimension)
        self.id_to_chunk: Dict[int, TextChunk] = {}

        if self.index_file.exists() and self.metadata_file.exists():
            self.load()

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """Embed and add chunks into FAISS index."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedding_engine.embed_documents(texts)
        emb_matrix = np.array(embeddings, dtype=np.float32)

        # Normalize vectors for cosine similarity via Inner Product
        self.faiss.normalize_L2(emb_matrix)

        start_idx = self.index.ntotal
        self.index.add(emb_matrix)

        for i, chunk in enumerate(chunks):
            self.id_to_chunk[start_idx + i] = chunk

        logger.info("Added %d vectors to FAISS index (Total: %d)", len(chunks), self.count())
        self.save()

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Query FAISS index with natural language query."""
        if self.count() == 0:
            return []

        query_emb = self.embedding_engine.embed_query(query)
        q_vec = np.array([query_emb], dtype=np.float32)
        self.faiss.normalize_L2(q_vec)

        # Retrieve a wider pool if metadata filtering is requested
        search_k = min(self.count(), top_k * 3 if filter_metadata else top_k)
        scores, indices = self.index.search(q_vec, search_k)

        results: List[VectorSearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx not in self.id_to_chunk:
                continue

            chunk = self.id_to_chunk[idx]

            # Apply metadata filtering
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    val = getattr(chunk, k, None) or chunk.metadata.get(k)
                    if val != v:
                        match = False
                        break
                if not match:
                    continue

            # In Inner Product with normalized vectors, score is cosine similarity [-1.0, 1.0]
            normalized_score = max(0.0, min(1.0, (float(score) + 1.0) / 2.0))

            results.append(VectorSearchResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=normalized_score,
                metadata=chunk.metadata,
                source_name=chunk.source_name,
                page_number=chunk.page_number,
                parent_id=chunk.parent_id,
                parent_text=chunk.parent_text,
            ))

            if len(results) >= top_k:
                break

        return results

    def count(self) -> int:
        return self.index.ntotal

    def clear(self) -> None:
        self.index = self.faiss.IndexFlatIP(self.dimension)
        self.id_to_chunk.clear()
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        logger.info("Cleared FAISS index and local metadata storage.")

    def save(self) -> None:
        """Persist FAISS index and chunk mappings to disk."""
        try:
            self.faiss.write_index(self.index, str(self.index_file))
            serializable_meta = {
                str(idx): chunk.to_dict() for idx, chunk in self.id_to_chunk.items()
            }
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(serializable_meta, f, indent=2)
            logger.info("Saved FAISS index to %s", self.index_file)
        except Exception as e:
            logger.error("Error saving FAISS index: %s", e)

    def load(self) -> None:
        """Load persisted FAISS index and chunk mappings from disk."""
        try:
            loaded_index = self.faiss.read_index(str(self.index_file))
            if loaded_index.d != self.dimension:
                logger.warning(
                    "Persisted FAISS index dimension (%d) does not match active model dimension (%d); resetting index.",
                    loaded_index.d,
                    self.dimension,
                )
                self.clear()
                return

            self.index = loaded_index
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.id_to_chunk = {}
            for k, v in data.items():
                chunk = TextChunk(
                    chunk_id=v["chunk_id"],
                    text=v["text"],
                    token_count=v["token_count"],
                    strategy=v["strategy"],
                    source_name=v["source_name"],
                    source_type=v["source_type"],
                    page_number=v["page_number"],
                    section_header=v.get("section_header", ""),
                    parent_id=v.get("parent_id"),
                    parent_text=v.get("parent_text"),
                    cluster_id=v.get("cluster_id"),
                    cluster_label=v.get("cluster_label"),
                    metadata=v.get("metadata", {}),
                )
                self.id_to_chunk[int(k)] = chunk
            logger.info("Loaded %d vectors from FAISS index %s", self.index.ntotal, self.index_file)
        except Exception as e:
            logger.error("Error loading FAISS index: %s", e)


# =============================================================================
# UNIFIED VECTOR STORE MANAGER
# =============================================================================

class VectorStoreManager:
    """Unified manager and factory interface for ChromaDB and FAISS stores."""

    def __init__(
        self,
        backend: Union[VectorDBType, str] = settings.DEFAULT_VECTOR_DB,
        embedding_engine: Optional[EmbeddingEngine] = None,
        embedding_model: Union[EmbeddingModelType, str] = settings.DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.backend = VectorDBType(backend) if isinstance(backend, str) else backend
        self.embedding_engine = embedding_engine or EmbeddingEngine(model_name=embedding_model)

        if self.backend == VectorDBType.CHROMADB:
            self.store: BaseVectorStore = ChromaVectorStore(embedding_engine=self.embedding_engine)
        elif self.backend == VectorDBType.FAISS:
            self.store = FAISSVectorStore(embedding_engine=self.embedding_engine)
        else:
            raise ValueError(f"Unsupported Vector DB backend: {self.backend}")

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """Index chunks into the active vector store."""
        self.store.add_chunks(chunks)

    def similarity_search(
        self,
        query: str,
        top_k: int = settings.DEFAULT_TOP_K,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Perform similarity search across active vector store."""
        return self.store.similarity_search(query=query, top_k=top_k, filter_metadata=filter_metadata)

    def count(self) -> int:
        return self.store.count()

    def clear(self) -> None:
        self.store.clear()
