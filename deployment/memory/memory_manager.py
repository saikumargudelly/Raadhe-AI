from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, storage_dir: str = "memory"):
        """
        Initialize the memory manager.
        
        Args:
            storage_dir: Directory to store memory files
        """
        self.storage_dir = storage_dir
        self.memories = {}
        self.create_storage_dir()
    
    def create_storage_dir(self) -> None:
        """Create the storage directory if it doesn't exist."""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            logger.info(f"Created memory storage directory: {self.storage_dir}")
    
    def add_memory(
        self,
        user_id: str,
        memory_type: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new memory for a user.
        
        Args:
            user_id: Unique identifier for the user
            memory_type: Type of memory (e.g., 'conversation', 'preference')
            content: The memory content
            metadata: Additional metadata about the memory
        """
        if user_id not in self.memories:
            self.memories[user_id] = []
        
        memory = {
            "type": memory_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.memories[user_id].append(memory)
        self._save_memories(user_id)
        logger.info(f"Added {memory_type} memory for user {user_id}")
    
    def get_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories for a user.
        
        Args:
            user_id: Unique identifier for the user
            memory_type: Filter by memory type
            limit: Maximum number of memories to return
            
        Returns:
            List of memories
        """
        if user_id not in self.memories:
            return []
        
        memories = self.memories[user_id]
        
        if memory_type:
            memories = [m for m in memories if m["type"] == memory_type]
        
        if limit:
            memories = memories[-limit:]
        
        return memories
    
    def clear_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None
    ) -> None:
        """
        Clear memories for a user.
        
        Args:
            user_id: Unique identifier for the user
            memory_type: Clear only memories of this type
        """
        if user_id not in self.memories:
            return
        
        if memory_type:
            self.memories[user_id] = [
                m for m in self.memories[user_id]
                if m["type"] != memory_type
            ]
        else:
            self.memories[user_id] = []
        
        self._save_memories(user_id)
        logger.info(f"Cleared memories for user {user_id}")
    
    def _save_memories(self, user_id: str) -> None:
        """
        Save memories to disk.
        
        Args:
            user_id: Unique identifier for the user
        """
        file_path = os.path.join(self.storage_dir, f"{user_id}.json")
        with open(file_path, "w") as f:
            json.dump(self.memories[user_id], f, indent=2)
    
    def _load_memories(self, user_id: str) -> None:
        """
        Load memories from disk.
        
        Args:
            user_id: Unique identifier for the user
        """
        file_path = os.path.join(self.storage_dir, f"{user_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                self.memories[user_id] = json.load(f)
    
    def get_memory_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of a user's memories.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing memory statistics
        """
        if user_id not in self.memories:
            return {
                "total_memories": 0,
                "memory_types": {}
            }
        
        memories = self.memories[user_id]
        memory_types = {}
        
        for memory in memories:
            memory_type = memory["type"]
            if memory_type not in memory_types:
                memory_types[memory_type] = 0
            memory_types[memory_type] += 1
        
        return {
            "total_memories": len(memories),
            "memory_types": memory_types
        } 