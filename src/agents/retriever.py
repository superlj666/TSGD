import logging
import argparse
import os
import sys
import numpy as np
import json
import threading
from collections import defaultdict
from typing import List, Optional, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from src.agents.base_agent import BaseAgent
from src.core.experience_pool import ExperiencePool, Experience
from src.tools.utils import Config, get_chat_model, DataLoader, suppress_external_logging, get_embeddings_model

class RetrievalAgent(BaseAgent):
    """
    Retrieval Agent (Pi_retr):
    Retrieves relevant experiences from the library using Problem-Based Retrieval.
    Loads a static index of Problem Embeddings and Metadata.
    """
    def __init__(self, experience_pool: ExperiencePool, config: Dict[str, Any] = None, embedding_model: str = None):
        super().__init__(config)
        self.experience_pool = experience_pool
        self.rerank = self.config.get("rerank", False)
        self.llm = self._init_llm("retriever") if self.rerank else None
        
        # Problem & Experience Embedding Index
        if embedding_model is None and self.config:
            embedding_model = self.config.get("embedding_model")
            
        logging.info(f"RetrievalAgent initialized. Config keys: {list(self.config.keys()) if self.config else 'None'}")
        logging.info(f"RetrievalAgent embedding_model: {embedding_model}")
        
        if embedding_model:
            self.embeddings_model = get_embeddings_model(embedding_model)
        else:
            self.embeddings_model = get_embeddings_model()
        self.problem_matrix: Optional[np.ndarray] = None
        self.experience_matrix: Optional[np.ndarray] = None
        self.problem_metadata: Dict[str, Dict[str, Any]] = {} # pid -> {problem, experience_ids, subject, level}
        self.problem_ids: List[str] = []
        self.experience_ids: List[str] = []

    def register_problem(self, item_id: str, problem: str, subject: str = None, level: str = None):
        """
        Registers a problem in the metadata for retrieval.
        Note: This does NOT update the embedding matrix immediately.
        """
        if item_id not in self.problem_metadata:
            self.problem_metadata[item_id] = {
                "problem": problem,
                "experience_ids": [],
                "subject": subject,
                "level": level
            }
            if item_id not in self.problem_ids:
                self.problem_ids.append(item_id)
            logging.debug(f"Registered problem {item_id} in retriever metadata.")

    def link_experience(self, exp_id: str, problem_id: str):
        """Links an experience ID to a registered problem ID."""
        if problem_id in self.problem_metadata:
            if exp_id not in self.problem_metadata[problem_id]["experience_ids"]:
                self.problem_metadata[problem_id]["experience_ids"].append(exp_id)
                logging.debug(f"Linked experience {exp_id} to problem {problem_id}")
        else:
            logging.warning(f"Attempted to link experience {exp_id} to unregistered problem {problem_id}")

    def save(self, output_dir: str):
        """Saves problem metadata to question_meta.json in the output directory."""
        os.makedirs(output_dir, exist_ok=True)
        meta_path = os.path.join(output_dir, "question_meta.json")
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.problem_metadata, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved problem metadata to {meta_path}")
        except Exception as e:
            logging.error(f"Failed to save problem metadata to {meta_path}: {e}")

    def load_data(self, embedding_path: str, input_dir: str = None, experience_embedding_path: str = None) -> None:
        """Loads the problem index, metadata, experience pool, and experience index."""
        try:
            # 1. Load Problem Matrix (Support .npz only)
            if os.path.exists(embedding_path):
                logging.info(f"Loading retrieval data from {embedding_path}...")
                data = np.load(embedding_path)
                self.problem_matrix = data["embeddings"]
                if "ids" in data:
                    self.problem_ids = data["ids"].tolist()
                    logging.info(f"Loaded problem IDs from .npz: {len(self.problem_ids)} items")
                logging.info(f"Loaded problem matrix from .npz: {self.problem_matrix.shape}")
            else:
                logging.warning(f"Problem index file not found: {embedding_path}")

            # 2. Load Problem Metadata
            search_dir = input_dir if input_dir else os.path.dirname(embedding_path)
            
            meta_path = os.path.join(search_dir, "question_meta_reg.json")
            if not os.path.exists(meta_path):
                meta_path = os.path.join(search_dir, "question_meta.json")
            
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.problem_metadata = json.load(f)
                
                # If we loaded IDs from NPZ, prioritize them or verify consistency
                loaded_meta_ids = list(self.problem_metadata.keys())
                if not self.problem_ids:
                    self.problem_ids = loaded_meta_ids
                    logging.info(f"Loaded problem metadata from {meta_path}: {len(self.problem_ids)} items")
                else:
                    # IDs already loaded from NPZ, just check count
                    if len(self.problem_ids) != len(loaded_meta_ids):
                         logging.warning(f"ID count mismatch! NPZ: {len(self.problem_ids)}, Meta: {len(loaded_meta_ids)}")
                    logging.info(f"Merged problem metadata from {meta_path}")
            else:
                logging.warning(f"Problem metadata file not found in {search_dir}")
            
            # 3. Load Experience Pool
            pool_path = os.path.join(search_dir, "experience_pool_reg.jsonl")
            if not os.path.exists(pool_path):
                pool_path = os.path.join(search_dir, "experience_pool.jsonl")
            
            if os.path.exists(pool_path):
                logging.info(f"Loading experience pool from {pool_path}")
                self.experience_pool.load(pool_path)
                self.experience_ids = [exp.id for exp in self.experience_pool.experiences]
                logging.info(f"Loaded {len(self.experience_pool.registry)} experiences into pool.")

                # Sync problem_metadata with loaded experience pool to fix stale IDs
                if self.problem_metadata:
                    logging.info("Syncing problem metadata with loaded experience pool...")
                    # 1. Build reverse index: Problem ID -> [Experience IDs] from the POOL
                    pool_index = defaultdict(set)
                    for exp in self.experience_pool.experiences:
                        sids = exp.source_id if isinstance(exp.source_id, list) else [exp.source_id]
                        for sid in sids:
                            if sid and sid != "N/A":
                                pool_index[str(sid)].add(exp.id)
                    
                    # 2. Update metadata
                    updates_count = 0
                    for pid in self.problem_metadata:
                        pid_str = str(pid)
                        current_links = set(self.problem_metadata[pid].get("experience_ids", []))
                        valid_links = pool_index.get(pid_str, set())
                        
                        if current_links != valid_links:
                            self.problem_metadata[pid]["experience_ids"] = list(valid_links)
                            updates_count += 1
                            
                    logging.info(f"Synced metadata: Updated experience links for {updates_count} problems.")

            else:
                logging.warning(f"No experience pool found in {search_dir}")
            
            # Fallback: Rebuild metadata from experience pool if missing
            if not self.problem_metadata and self.experience_pool.registry:
                if self.problem_ids:
                    logging.info("Problem metadata missing. Rebuilding from problem_ids and Experience Pool...")
                    rebuilt_meta = {}
                    
                    # Pre-index experiences by source_id for efficiency
                    exp_by_source = defaultdict(list)
                    for exp in self.experience_pool.experiences:
                        sids = exp.source_id if isinstance(exp.source_id, list) else [exp.source_id]
                        for sid in sids:
                            if sid and sid != "N/A":
                                exp_by_source[str(sid)].append(exp)
                    
                    # Build metadata using problem_ids as keys
                    for pid in self.problem_ids:
                        pid_str = str(pid)
                        rebuilt_meta[pid_str] = {
                            "problem": "N/A (Rebuilt)",
                            "experience_ids": [],
                            "subject": "N/A",
                            "level": "N/A"
                        }
                        
                        if pid_str in exp_by_source:
                            matches = exp_by_source[pid_str]
                            rebuilt_meta[pid_str]["experience_ids"] = [e.id for e in matches]
                            if matches:
                                # Inherit metadata from the first matching experience
                                rebuilt_meta[pid_str]["subject"] = matches[0].subject
                                rebuilt_meta[pid_str]["level"] = matches[0].level

                    self.problem_metadata = rebuilt_meta
                    logging.info(f"Rebuilt metadata for {len(self.problem_ids)} problems.")

                    # Save the generated index file
                    meta_save_path = os.path.join(search_dir, "question_meta.json")
                    try:
                        with open(meta_save_path, 'w', encoding='utf-8') as f:
                            json.dump(self.problem_metadata, f, indent=2, ensure_ascii=False)
                        logging.info(f"Saved rebuilt metadata to {meta_save_path}")
                    except Exception as e:
                        logging.error(f"Failed to save rebuilt metadata: {e}")
                else:
                    logging.warning("Problem metadata missing and no problem_ids available from NPZ. Cannot rebuild metadata.")

            # 4. Load Experience Matrix (Optional)
            if not experience_embedding_path:
                experience_embedding_path = os.path.join(search_dir, "experience_idx.npz")
            
            if os.path.exists(experience_embedding_path):
                data = np.load(experience_embedding_path)
                if "embeddings" in data:
                    self.experience_matrix = data["embeddings"]
                elif "arr_0" in data:
                    self.experience_matrix = data["arr_0"]
                else:
                    self.experience_matrix = None
                    logging.warning(f"Could not find 'embeddings' or 'arr_0' in {experience_embedding_path}")

                if self.experience_matrix is not None:
                     logging.info(f"Loaded experience matrix: {self.experience_matrix.shape}")
            else:
                logging.info(f"Experience index file not found: {experience_embedding_path}")

        except Exception as e:
            logging.error(f"Failed to load retrieval data: {e}")
            raise e

    def run(self, query: str, k: int = 5, item_id: str = "N/A", subject: Optional[str] = None, difficulty: Optional[str] = None, mode: str = "problem") -> List[Experience]:
        """
        Retrieves relevant experiences.
        mode: 'problem' (default) or 'random'
        """
        # Use pool configuration if available, otherwise fallback to method arguments
        k = getattr(self.experience_pool, 'retrieval_top_k', k) or k
        
        # Fix: Allow similarity_threshold to be 0.0
        similarity_threshold = getattr(self.experience_pool, 'similarity_threshold')
        if similarity_threshold is None:
            similarity_threshold = 0.4
        
        if not self.experience_pool.registry:
            logging.info(f"[{item_id}] Experience pool is empty. Skipping retrieval.")
            return []

        # --- GLOBAL FILTER: Filter experiences by subject/difficulty FIRST ---
        all_exps = list(self.experience_pool.registry.values())
        valid_exps = []
        valid_exp_ids = set()

        for exp in all_exps:
            # Filter by subject
            if subject and exp.subject and exp.subject != subject:
                continue
            # Filter by difficulty
            if difficulty and exp.level and str(exp.level) != str(difficulty):
                continue
            
            valid_exps.append(exp)
            valid_exp_ids.add(exp.id)

        logging.info(f"ALL EXP: [{len(all_exps)}] --- VALID EXP: [{len(valid_exps)}]")
        

        if not valid_exps:
            logging.info(f"[{item_id}] No experiences match filters (Subject={subject}, Difficulty={difficulty}).")
            return []

        # --- MODE: Random Retrieval ---
        if mode == "random":
            import random
            logging.info(f"[{item_id}] Retrieving experiences (Mode=random)")
            
            if len(valid_exps) <= k:
                candidates = valid_exps
            else:
                candidates = random.sample(valid_exps, k)
            
            for exp in candidates:
                logging.info(f"[{item_id}] Randomly Retrieved Exp: {exp.id}")
            return candidates

        query = self._normalize_query(query)
        logging.info(f"[{item_id}] Retrieving experiences (Mode={mode}, Subject={subject or 'All'}, Difficulty={difficulty or 'All'})")
        
        try:
            # 1. Embed Query
            query_emb = self.embeddings_model.embed_query(query)
            query_vec = np.array(query_emb).reshape(1, -1)
            
            exp_candidates = [] # List of (score, Experience)
            seen_exp_ids = set()

            # --- MODE: Problem-Based Retrieval ---
            if mode == "problem":
                if self.problem_matrix is not None and len(self.problem_matrix) > 0:
                    # Validate problem_ids
                    if not self.problem_ids:
                        logging.warning(f"[{item_id}] Problem IDs missing (metadata not loaded?), skipping problem-based retrieval.")
                    else:
                        if len(self.problem_matrix) != len(self.problem_ids):
                            active_matrix = self.problem_matrix[:len(self.problem_ids)]
                        else:
                            active_matrix = self.problem_matrix
                        
                        if len(active_matrix) == 0:
                            logging.warning(f"[{item_id}] Active problem matrix is empty, skipping.")
                        else:
                            # 1. Filter indices by subject/difficulty BEFORE vector search (Problem Level Optimization)
                            valid_indices = []
                            for i, pid in enumerate(self.problem_ids):
                                if i >= len(active_matrix): break
                                
                                prob_meta = self.problem_metadata.get(pid, {})
                                
                                # Filter by subject (Problem Metadata)
                                if subject and prob_meta.get("subject") and prob_meta.get("subject") != subject:
                                    continue
                                
                                # Filter by difficulty (Problem Metadata)
                                if difficulty and prob_meta.get("level") and str(prob_meta.get("level")) != str(difficulty):
                                    continue
                                    
                                valid_indices.append(i)

                            if not valid_indices:
                                logging.info(f"[{item_id}] No problems match filters (Subject={subject}, Difficulty={difficulty}).")
                            else:
                                # 2. Perform vector search only on filtered subset
                                filtered_matrix = active_matrix[valid_indices]
                                sims = cosine_similarity(query_vec, filtered_matrix)[0]
                                
                                # 3. Sort and Select
                                top_n_local = min(len(sims), max(5, k * 2))
                                top_local_indices = np.argsort(sims)[::-1][:top_n_local]
                                
                                logging.info(f"[{item_id}]   - Searching via similar problems... (Threshold: {similarity_threshold})")
                                
                                for local_idx in top_local_indices:
                                    score = sims[local_idx]
                                    original_idx = valid_indices[local_idx] # Map back to global index
                                    pid = self.problem_ids[original_idx]
                                    
                                    if score < similarity_threshold:
                                        logging.debug(f"[{item_id}] Problem {pid} score {score:.4f} < {similarity_threshold}, skipping.")
                                        continue
                                        
                                    prob_meta = self.problem_metadata.get(pid, {})
                                    linked_exp_ids = prob_meta.get("experience_ids", [])
                                    
                                    if not linked_exp_ids:
                                        logging.debug(f"[{item_id}] Problem {pid} (score {score:.4f}) has no linked experiences.")

                                    for eid in linked_exp_ids:
                                        # Only check if eid is in the pre-filtered valid_exp_ids set
                                        if eid in valid_exp_ids:
                                            if eid not in seen_exp_ids:
                                                exp = self.experience_pool.registry[eid]
                                                
                                                # Problem similarity score acts as weight
                                                exp_candidates.append((score, exp))
                                                seen_exp_ids.add(eid)
                                        else:
                                            # Debug logging for skipped experiences
                                            if eid not in self.experience_pool.registry:
                                                logging.debug(f"[{item_id}] Linked exp {eid} missing from pool.")
                                            else:
                                                logging.debug(f"[{item_id}] Linked exp {eid} filtered by subject/level.")
                                
                                logging.info(f"#linked_exp_ids [{len(linked_exp_ids)}]  --- #prob_meta [{len(prob_meta)}] prob_meta")
                else:
                    logging.warning(f"[{item_id}] Problem matrix missing, skipping problem-based retrieval.")


            # Sort and filter
            exp_candidates.sort(key=lambda x: x[0], reverse=True)
            
            for score, exp in exp_candidates[:k]:
                logging.info(f"[{item_id}] Retrieved Exp: {exp.id}, Similarity: {score:.4f}")

            candidates = [exp for score, exp in exp_candidates][:k]
            
        except Exception as e:
            logging.error(f"Retrieval failed: {e}")
            candidates = []

        if not candidates:
            logging.info(f"[{item_id}] No candidates found.")
            return []

        # Agentic Rerank (optional)
        if not self.rerank or not self.llm:
            return candidates
            
        # Agentic Rerank logic remains same...
        reranked = []
        for exp in candidates:
            system_prompt = Config.RETRIEVER_RERANK_SYSTEM_PROMPT
            prompt = Config.RETRIEVER_RERANK_USER_PROMPT.format(query=query, condition=exp.condition)
            try:
                response = self.llm.invoke([
                    ("system", system_prompt),
                    ("user", prompt)
                ])
                if "yes" in response.content.lower():
                    reranked.append(exp)
            except Exception as e:
                logging.error(f"Rerank failed for exp {exp.id}: {e}")
                reranked.append(exp) 
        
        return reranked

    def _normalize_query(self, query: str) -> str:
        query = query.strip()
        query = query.replace("\\text{", "").replace("}", "")
        query = query.replace("\\quad", " ").replace("\\qquad", " ")
        return query

def calculate_metrics(retrieved_ids: List[str], ground_truth_ids: List[str]):
    if not ground_truth_ids:
        return 0.0, 0.0, 0.0
    
    retrieved_set = set(retrieved_ids)
    gt_set = set(ground_truth_ids)
    
    tp = len(retrieved_set.intersection(gt_set))
    precision = tp / len(retrieved_set) if retrieved_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def evaluate_retriever(args: argparse.Namespace):
    # 1. Initialize Pool & Agent
    pool = ExperiencePool()
    pool.similarity_threshold = getattr(args, 'similarity_threshold', 0.4)
    pool.retrieval_top_k = args.k

    agent = RetrievalAgent(pool, config={"rerank": args.rerank}, embedding_model=args.embedding_model)
    
    # 2. Load Data
    embedding_path = args.problem_embedding_path
    if not embedding_path and args.input_dir:
        embedding_path = os.path.join(args.input_dir, "question_indexing.npz")
    
    exp_embedding_path = getattr(args, 'experience_embedding_path', None)
    
    agent.load_data(
        embedding_path=embedding_path, 
        input_dir=args.input_dir,
        experience_embedding_path=exp_embedding_path
    )
    
    # Print Hyperparameters
    print("\n" + "="*70)
    print("Retrieval Hyperparameters:")
    print(f"  - Embedding Model: {args.embedding_model or 'default'}")
    print(f"  - Retrieval Mode: {args.mode}")
    print(f"  - Top-K: {args.k}")
    print(f"  - Similarity Threshold: {pool.similarity_threshold}")
    print(f"  - Rerank: {args.rerank}")
    print("="*70 + "\n")

    # 3. Load Queries
    queries = DataLoader.load_dataset(args.query_path)
    if args.max_samples:
        queries = queries[:args.max_samples]

    results = []
    
    for i, item in enumerate(queries):
        query_text = item.get("problem", "")
        item_id = item.get("item_id", f"query_{i}")
        # Get Ground Truth: Try item first, then metadata
        gt_ids = item.get("ground_truth_experience_ids", [])
        if not gt_ids and item_id in agent.problem_metadata:
            gt_ids = agent.problem_metadata[item_id].get("experience_ids", [])
        
        subject = item.get("subject")

        # Run Retrieval
        retrieved_exps = agent.run(
            query=query_text, 
            k=args.k, 
            item_id=item_id, 
            subject=subject,
            mode=args.mode
        )
        retrieved_ids = [exp.id for exp in retrieved_exps]

        # Calculate Metrics
        p, r, f1 = calculate_metrics(retrieved_ids, gt_ids)
        results.append({"p": p, "r": r, "f1": f1})

        print(f"[{item_id}] P: {p:.2f} | R: {r:.2f} | F1: {f1:.2f} | Found: {len(retrieved_ids)} | GT: {len(gt_ids)}")

    # 4. Aggregate Results
    if results:
        avg_p = sum(r["p"] for r in results) / len(results)
        avg_r = sum(r["r"] for r in results) / len(results)
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        
        print("\n" + "="*70)
        print("FINAL RESULTS:")
        print(f"  - Average Precision: {avg_p:.4f}")
        print(f"  - Average Recall:    {avg_r:.4f}")
        print(f"  - Average F1 Score:  {avg_f1:.4f}")
        print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--problem_embedding_path", type=str)
    parser.add_argument("--experience_embedding_path", type=str)
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-large")
    parser.add_argument("--mode", type=str, default="problem", choices=["problem", "condition", "problem"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--similarity_threshold", type=float, default=0.4)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--max_samples", type=int)
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    suppress_external_logging()
    
    evaluate_retriever(args)
