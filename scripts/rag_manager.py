import torch
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import logging
import json
import os
from pathlib import Path
from cache_manager import CacheManager
from logger_config import get_logger
from error_handler import safe_execute
from performance_monitor import PerformanceMonitor
import time

logger = logging.getLogger(__name__)

class RAGManager:
    """
    Enhanced RAG (Retrieval-Augmented Generation) manager for improved conversations.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_context_length: int = 2048,
        context_weight: float = 0.7,
        cache_dir: str = "cache"
    ):
        """
        Initialize the RAG manager.
        
        Args:
            embedding_model: Model to use for embeddings
            max_context_length: Maximum length of context to use
            context_weight: Weight for context in response generation
            cache_dir: Directory to store cache
        """
        self.embedding_model = embedding_model
        self.max_context_length = max_context_length
        self.context_weight = context_weight
        self.cache_dir = Path(cache_dir)
        
        # Initialize memory components
        self.importance_scores = {}
        self.context_window = []
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(embedding_model).to(self.device)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.contexts = []
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache if available
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load existing conversations and embeddings from cache."""
        try:
            # Load conversations
            conv_path = self.cache_dir / "conversations.json"
            if conv_path.exists():
                with open(conv_path, "r") as f:
                    self.conversations = json.load(f)
            else:
                self.conversations = []
            
            # Load embeddings
            emb_path = self.cache_dir / "embeddings.npy"
            if emb_path.exists():
                self.embeddings = np.load(emb_path)
                self._build_index()
            else:
                self.embeddings = None
                
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            self.conversations = []
            self.embeddings = None
            
    def _save_cache(self) -> None:
        """Save conversations and embeddings to cache."""
        try:
            # Save conversations
            conv_path = self.cache_dir / "conversations.json"
            with open(conv_path, "w") as f:
                json.dump(self.conversations, f)
            
            # Save embeddings
            if self.embeddings is not None:
                emb_path = self.cache_dir / "embeddings.npy"
                np.save(emb_path, self.embeddings)
                
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
            
    def _build_index(self) -> None:
        """Build FAISS index from embeddings."""
        if self.embeddings is not None and len(self.embeddings) > 0:
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings.astype('float32'))
            
    def add_conversation(self, conversation: List[Dict[str, str]]) -> None:
        """Add a new conversation to the cache."""
        try:
            # Extract text from conversation
            text = " ".join([msg["content"] for msg in conversation])
            
            # Generate embedding
            embedding = self.encoder.encode([text])[0]
            
            # Update conversations and embeddings
            self.conversations.append(conversation)
            if self.embeddings is None:
                self.embeddings = embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
            
            # Rebuild index
            self._build_index()
            
            # Save to cache
            self._save_cache()
            
        except Exception as e:
            logger.error(f"Error adding conversation: {str(e)}")
            
    def get_relevant_context(self, query: str, k: int = 3, threshold: float = 0.7) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Get relevant context for the current query.
        
        Args:
            query: Current user query
            k: Number of relevant conversations to retrieve
            threshold: Similarity threshold
            
        Returns:
            Tuple of (relevant conversations, similarity scores)
        """
        try:
            if not query or not isinstance(query, str):
                logger.warning("Invalid query provided")
                return [], []
                
            if self.index is None or len(self.conversations) == 0:
                return [], []
                
            # Generate query embedding
            query_embedding = self.encoder.encode([query])[0].reshape(1, -1).astype('float32')
            
            # Search for similar conversations
            distances, indices = self.index.search(query_embedding, min(k, len(self.conversations)))
            
            # Return relevant conversations with scores
            relevant_contexts = []
            scores = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.conversations):
                    score = float(1 / (1 + distance))  # Convert distance to similarity score
                    if score >= threshold:
                        relevant_contexts.append({
                            "conversation": self.conversations[idx],
                            "relevance_score": score
                        })
                        scores.append(score)
            
            return relevant_contexts, scores
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return [], []
            
    def enhance_prompt(self, prompt: str, history: List[Dict[str, str]] = None) -> str:
        """Enhance the prompt with relevant context from the cache."""
        try:
            # Get relevant context
            relevant_contexts, scores = self.get_relevant_context(prompt)
            
            # If no relevant context, return original prompt
            if not relevant_contexts:
                return prompt
                
            # Build enhanced prompt
            enhanced_prompt = "Previous relevant conversations:\n\n"
            for ctx, score in zip(relevant_contexts, scores):
                conv = ctx["conversation"]
                enhanced_prompt += f"Conversation (relevance: {score:.2f}):\n"
                for msg in conv:
                    enhanced_prompt += f"{msg['role']}: {msg['content']}\n"
                enhanced_prompt += "\n"
                
            enhanced_prompt += f"\nCurrent conversation:\n"
            if history:
                for msg in history:
                    enhanced_prompt += f"{msg['role']}: {msg['content']}\n"
            enhanced_prompt += f"user: {prompt}\n"
            enhanced_prompt += "assistant:"
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error enhancing prompt: {str(e)}")
            return prompt
            
    def clear_cache(self) -> None:
        """Clear the conversation cache."""
        try:
            self.conversations = []
            self.embeddings = None
            self.index = None
            self._save_cache()
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        return {
            "total_memories": len(self.conversations),
            "context_window_size": len(self.context_window),
            "average_importance": np.mean(list(self.importance_scores.values())) if self.importance_scores else 0,
            "max_importance": max(self.importance_scores.values()) if self.importance_scores else 0,
            "min_importance": min(self.importance_scores.values()) if self.importance_scores else 0
        }
    
    def add_context(self, context: str) -> None:
        """
        Add a new context to the retrieval system.
        
        Args:
            context: Text context to add
        """
        if not context.strip():
            return
            
        # Encode the context
        with torch.no_grad():
            embedding = self.encoder.encode([context], convert_to_tensor=True)
            embedding = embedding.cpu().numpy()
            
        # Add to FAISS index
        self.index.add(embedding)
        self.contexts.append(context)
        
    def retrieve_relevant_contexts(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve relevant contexts for a given query.
        
        Args:
            query: The query text
            k: Number of contexts to retrieve
            
        Returns:
            List of relevant context strings
        """
        if not self.contexts:
            return []
            
        start_time = time.time()
        
        # Encode the query
        with torch.no_grad():
            query_embedding = self.encoder.encode([query], convert_to_tensor=True)
            query_embedding = query_embedding.cpu().numpy()
            
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, min(k, len(self.contexts)))
        
        # Get relevant contexts
        relevant_contexts = [self.contexts[i] for i in indices[0]]
        
        # Record performance metrics if monitor is available
        if self.performance_monitor:
            retrieval_time = time.time() - start_time
            self.performance_monitor.record_context_retrieval_time(retrieval_time)
            
        return relevant_contexts
        
    def generate_enhanced_prompt(self, query: str, conversation_history: List[Dict], k: int = 3) -> str:
        """
        Generate an enhanced prompt using retrieved contexts.
        
        Args:
            query: The user's query
            conversation_history: List of previous conversation messages
            k: Number of contexts to retrieve
            
        Returns:
            Enhanced prompt string
        """
        # Retrieve relevant contexts
        relevant_contexts = self.retrieve_relevant_contexts(query, k)
        
        # Build the enhanced prompt
        prompt_parts = []
        
        # Add system message
        prompt_parts.append("You are a helpful AI assistant. Use the following context to inform your response:")
        
        # Add relevant contexts
        if relevant_contexts:
            prompt_parts.append("\nRelevant context:")
            for i, context in enumerate(relevant_contexts, 1):
                prompt_parts.append(f"{i}. {context}")
                
        # Add conversation history
        if conversation_history:
            prompt_parts.append("\nConversation history:")
            for message in conversation_history[-5:]:  # Last 5 messages for context
                role = message.get("role", "user")
                content = message.get("content", "")
                prompt_parts.append(f"{role}: {content}")
                
        # Add current query
        prompt_parts.append(f"\nCurrent query: {query}")
        
        return "\n".join(prompt_parts)
        
    def clear_contexts(self) -> None:
        """Clear all stored contexts."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.contexts = []
        logger.info("All contexts cleared") 