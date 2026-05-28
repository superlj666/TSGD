import logging
import numpy as np
import json
import os
import uuid
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from src.tools.utils import Config, get_embeddings_model

import threading

class Experience:
    """
    Represents a single learned experience or rule in the library.
    e = { condition -> strategy }
    """
    def __init__(
        self, 
        condition: str, 
        strategy: str, 
        id: Optional[str] = None,
        source_id: Optional[List[str]] = None,
        created_by_agent: str = "Pi_init",
        subject: Optional[str] = None,
        level: Optional[str] = None,
        warning: str = ""
    ):
        if id:
            self.id = id
        else:
            # Generate a short 8-character hash based on content and timestamp
            content = f"{condition}{strategy}{time.time()}".encode('utf-8')
            self.id = hashlib.md5(content).hexdigest()[:8]
            
        self.condition = condition
        self.strategy = strategy
        self.subject = subject
        self.level = level
        self.warning = warning
        
        # Tracking for Utility Score
        if source_id is None:
            self.source_id = ["N/A"]
        elif isinstance(source_id, list):
            self.source_id = [str(i) for i in source_id if i]
            if not self.source_id:
                self.source_id = ["N/A"]
        else:
            sid = str(source_id)
            self.source_id = [sid] if sid else ["N/A"]
        
        self.created_by_agent = created_by_agent
        self.success_count: int = 0
        self.usage_count: int = 0
        self.created_at: float = time.time()
        self.utility_score: float = 0.0

        # Concurrency Control
        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._readers_count = 0

    def is_write_locked(self) -> bool:
        """Check if the experience is currently locked for writing."""
        return self._write_lock.locked()

    def acquire_read_lock(self) -> bool:
        """
        Acquire a read lock (increment reader count).
        Returns True if successful, False if write locked.
        """
        if self._write_lock.locked():
            return False
            
        with self._read_lock:
            self._readers_count += 1
        return True

    def release_read_lock(self):
        """Release a read lock (decrement reader count)."""
        with self._read_lock:
            if self._readers_count > 0:
                self._readers_count -= 1

    def acquire_write_lock(self, timeout: float = -1) -> bool:
        """
        Acquire a write lock (exclusive access).
        Requires no active readers.
        """
        if not self._write_lock.acquire(timeout=timeout):
            return False
            
        # Once we have the write lock, wait for readers to drain?
        # Or just fail if there are readers?
        # A simple check is not enough for strict consistency, but for this heuristic:
        # If we have write lock, new readers are blocked (by acquire_read_lock check).
        # We just need to wait for existing readers.
        
        # Simple spin wait for existing readers
        start_wait = time.time()
        while True:
            with self._read_lock:
                if self._readers_count == 0:
                    return True
            
            if timeout > 0 and (time.time() - start_wait) > timeout:
                self._write_lock.release()
                return False
                
            time.sleep(0.01)

    def release_write_lock(self):
        """Release the write lock."""
        self._write_lock.release()


    def calculate_utility(self, lam: Optional[float] = None, epsilon: Optional[float] = None) -> float:
        """
        Utility = (Success Count / (Usage Count + epsilon)) + lambda * Recency
        Recency is normalized based on current time.
        """
        lam = lam if lam is not None else Config.REGULARIZATION_LAMBDA
        epsilon = epsilon if epsilon is not None else Config.REGULARIZATION_EPSILON
        
        success_rate = self.success_count / (self.usage_count + epsilon)
        
        # Normalized recency: higher for more recent experiences.
        # current_at / current_time gives a value close to 1 for recent items.
        recency = self.created_at / time.time() 
        
        self.utility_score = success_rate + lam * recency
        return self.utility_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, 
            "condition": self.condition, 
            "strategy": self.strategy,
            "subject": self.subject,
            "level": self.level,
            "warning": self.warning,
            "source_id": self.source_id,
            "created_by_agent": self.created_by_agent,
            "success_count": self.success_count,
            "usage_count": self.usage_count,
            "created_at": datetime.fromtimestamp(self.created_at).strftime('%Y-%m-%d %H:%M:%S'),
            "utility_score": self.utility_score
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Experience':
        condition = data.get('condition') or data.get('problem') or ""
        strategy = data.get('strategy') or data.get('reflection') or data.get('solution') or ""
        item_id = data.get('id') or data.get('exp_id')
        subject = data.get('subject')
        level = data.get('level') or data.get('difficulty')
        warning = data.get('warning', "")
        
        # Robust source_id handling
        source_id = data.get('source_id', [])
        if source_id == "N/A":
            source_id = ["N/A"]
        elif not isinstance(source_id, list):
            source_id = [str(source_id)]
        
        # Filter out empty strings but keep "N/A" if it's the only one
        source_id = [str(i) for i in source_id if i]
        if not source_id:
            source_id = []
        
        exp = Experience(
            condition=condition, 
            strategy=strategy, 
            id=item_id,
            source_id=source_id,
            created_by_agent=data.get('created_by_agent', "Pi_init"),
            subject=subject,
            level=level,
            warning=warning
        )
        exp.success_count = data.get('success_count', 0)
        exp.usage_count = data.get('usage_count', 0)
        
        # Handle created_at (string or float)
        created_at = data.get('created_at', time.time())
        if isinstance(created_at, str):
            try:
                exp.created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S').timestamp()
            except ValueError:
                exp.created_at = time.time()
        else:
            exp.created_at = float(created_at)
            
        exp.utility_score = data.get('utility_score', 0.0)
        return exp

class ExperiencePool:
    """
    Manages the experience library using a Registry Pattern.
    - Source of Truth (registry): Full Experience objects stored in memory.
    - Search Index (vector_index): condition embeddings mapped to IDs.
    """
    def __init__(
        self, 
        max_pool_size: int = 300, 
        retrieval_top_k: int = 10, 
        similarity_threshold: float = 0.2
    ):
        # Source of Truth: ID -> Experience
        self.registry: Dict[str, Experience] = {}
        self.max_pool_size = max_pool_size
        self.retrieval_top_k = retrieval_top_k
        self.similarity_threshold = similarity_threshold
        self.embeddings_model = get_embeddings_model()

    @property
    def experiences(self) -> List[Experience]:
        """Returns all experiences as a list."""
        return list(self.registry.values())

    def add(self, experience: Experience) -> None:
        """Adds a new experience to the library."""
        # 1. Update Source of Truth
        self.registry[experience.id] = experience

    def batch_add(self, experiences: List[Experience]) -> None:
        """Adds multiple experiences."""
        if not experiences:
            return

        # 1. Update Source of Truth
        for exp in experiences:
            self.registry[exp.id] = exp

    def delete(self, exp_id: str) -> None:
        """Removes an experience by ID from registry."""
        if exp_id in self.registry:
            del self.registry[exp_id]

    def update(self, exp_id: str, condition: Optional[str] = None, strategy: Optional[str] = None, warning: Optional[str] = None) -> bool:
        """Updates an existing experience."""
        if exp_id not in self.registry:
            return False
            
        exp = self.registry[exp_id]
        if condition:
            exp.condition = condition
        if strategy:
            exp.strategy = strategy
        if warning:
            exp.warning = warning
        return True

    def merge(self, id1: str, id2: str, new_content: str) -> bool:
        """Merges two experiences into a new one and removes the originals."""
        exp1 = self.registry.get(id1)
        exp2 = self.registry.get(id2)
        if exp1 and exp2:
            merged_source_ids = list(set(exp1.source_id + exp2.source_id))
            
            # Merge warnings if they exist
            w1 = getattr(exp1, "warning", "")
            w2 = getattr(exp2, "warning", "")
            if w1 and w2:
                new_warning = f"{w1} | {w2}"
            else:
                new_warning = w1 or w2 or ""

            new_exp = Experience(
                condition=f"{exp1.condition} | {exp2.condition}",
                strategy=new_content,
                source_id=merged_source_ids,
                created_by_agent="Pi_reg",
                warning=new_warning
            )
            new_exp.usage_count = exp1.usage_count + exp2.usage_count
            new_exp.success_count = exp1.success_count + exp2.success_count
            
            self.delete(id1)
            self.delete(id2)
            self.add(new_exp)
            return True
        return False

    def get_by_id(self, exp_id: str) -> Optional[Experience]:
        """Returns an experience from the registry by ID."""
        return self.registry.get(exp_id)

    def _get_condition_embedding(self, condition: str) -> List[float]:
        """Generates embedding for a condition string."""
        try:
            return self.embeddings_model.embed_query(condition)
        except Exception as e:
            logging.error(f"Embedding failed for condition: {e}")
            return [0.0] * 1536

    def save(self, path: str = "experience_pool.jsonl", save_index: bool = False) -> None:
        """
        Saves the library to a JSONL file (Source of Truth).
        If path is a directory, saves to path/experience_pool.jsonl.
        """
        if os.path.isdir(path):
            path = os.path.join(path, "experience_pool.jsonl")
        
        with open(path, 'w', encoding='utf-8') as f:
            for exp in self.experiences:
                f.write(json.dumps(exp.to_dict(), ensure_ascii=False) + '\n')
            
    def load(self, path: str = "experience_pool.jsonl", rebuild_index: bool = False) -> None:
        """
        Loads the library from JSON or JSONL.
        If path is a directory, loads from path/experience_pool.jsonl.
        """
        if os.path.isdir(path):
            path = os.path.join(path, "experience_pool.jsonl")
            
        if not os.path.exists(path):
            return

        # 1. Load Registry
        self.registry = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('['):
                    # Standard JSON array
                    data_list = json.loads(content)
                    for item in data_list:
                        exp = Experience.from_dict(item)
                        self.registry[exp.id] = exp
                else:
                    # JSONL format
                    f.seek(0)
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            exp = Experience.from_dict(data)
                            self.registry[exp.id] = exp
        except Exception as e:
            logging.error(f"Failed to load experience pool from {path}: {e}")
            return


