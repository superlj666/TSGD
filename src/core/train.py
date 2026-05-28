import logging
import json
import os
import random
import time
from datetime import datetime
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.experience_pool import ExperiencePool, Experience
from src.agents.solver import SolverAgent
from src.agents.optimizer import OptimizerAgent
from src.agents.regularizer import RegularizerAgent
from src.agents.retriever import RetrievalAgent
from src.agents.initializer import InitAgent
from src.tools.utils import Config, LOG_LOCK, get_chat_model
from src.tools.grader import extract_answer, math_equal

import queue
import threading

class Trainer:
    """
    Orchestrates the TSGD training loop.
    Flow: Seeding -> Train Loop (Parallel Problem Processing) -> Regularization
    """
    def __init__(
        self, 
        experience_pool: ExperiencePool,
        solver: SolverAgent,
        optimizer: OptimizerAgent,
        regularizer: RegularizerAgent,
        retrieval: RetrievalAgent,
        initializer: InitAgent,
        val_data: List[Dict] = None,
        config: Dict[str, Any] = None
    ):
        self.experience_pool = experience_pool
        self.solver = solver
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.retrieval = retrieval
        self.initializer = initializer
        self.val_data = val_data or []
        self.test_data = config.get("test_data", []) if config else []
        self.val_embeddings = None
        self.config = config or {}
        self.enable_dual_verification = self.config.get("enable_dual_verification", False)
        self.enable_regularizer = self.config.get("enable_regularizer", False)
        self.max_optimization_steps = self.config.get("epochs", Config.MAX_OPTIMIZATION_STEPS)
        self.llm = get_chat_model()
        self.max_workers = self.config.get("max_workers", Config.MAX_WORKERS)
        self.problem_pool = queue.Queue()

    def _save_pool_snapshot(self, reason: str = "update"):
        """Saves a timestamped snapshot of the experience pool."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            intermediate_dir = os.path.join(self.config.get("experience_dir", "experiments"), "intermediate")
            os.makedirs(intermediate_dir, exist_ok=True)
            filename = f"pool_{timestamp}_{reason}.jsonl"
            path = os.path.join(intermediate_dir, filename)
            self.experience_pool.save(path)
            logging.info(f"Saved pool snapshot to {path}")
        except Exception as e:
            logging.error(f"Failed to save pool snapshot: {e}")

    def seed(self, seed_data: List[Dict]):
        """
        Seeding phase: Initialize the experience pool using InitAgent.
        """
        if len(self.experience_pool.experiences) > 0:
            logging.info(f"Experience pool already has {len(self.experience_pool.experiences)} items. Skipping seeding.")
            return

        logging.info("Starting Seeding phase...")
        
        # Register source problems for retrieval first
        for item in seed_data:
            self.retrieval.register_problem(
                item.get("item_id", "N/A"), 
                item.get("problem", ""), 
                item.get("subject"), 
                item.get("level") or item.get("difficulty")
            )

        # Batch run initialization
        exps = self.initializer.batch_run(seed_data, max_workers=self.max_workers)
        
        for exp in exps:
            self.experience_pool.add(exp)
            for sid in exp.source_id:
                self.retrieval.link_experience(exp.id, sid)
        
        logging.info(f"Seeding complete. Initial pool size: {len(self.experience_pool.experiences)}")
        self._save_pool_snapshot("seed")

    def train(self, train_data: List[Dict]):
        """
        Main training loop implementing Threaded TSGD.
        """
        # Pre-compute validation embeddings once at the start
        if self.val_data:
            self._ensure_val_embeddings()

        logging.info(f"Starting Threaded TSGD Training Loop: {len(train_data)} samples, {self.max_optimization_steps} steps.")
        
        # Setup for tracking
        global_step = 0
        intermediate_dir = os.path.join(self.config.get("experience_dir", "experiments"), "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        metrics_file = os.path.join(self.config.get("experience_dir", "experiments"), "training_metrics.jsonl")

        for step in range(self.max_optimization_steps):
            logging.info(f"--- Epoch {step + 1}/{self.max_optimization_steps} ---")
            
            # Fill problem pool
            # Shuffle training data
            random.shuffle(train_data)
            self.problem_pool = queue.Queue()
            for item in train_data:
                self.problem_pool.put(item)
                
            logging.info(f"Initialized Problem Pool with {self.problem_pool.qsize()} items.")
            
            # Start Worker Threads
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # We submit 'max_workers' persistent tasks that pull from queue
                futures = []
                for i in range(self.max_workers):
                    futures.append(executor.submit(self._worker_task, i))
                
                # Wait for all workers to finish (when queue is empty)
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        logging.error(f"Worker thread failed: {e}")
                
            # 4. Save Intermediate Pool
            try:
                pool_path = os.path.join(intermediate_dir, f"pool_epoch_{step + 1}.jsonl")
                self.experience_pool.save(pool_path)
            except Exception as e:
                logging.error(f"Failed to save intermediate pool: {e}")

            # Regularization Phase (Moved to outside the item loop)
            if self.enable_regularizer:
                logging.info(f"--- Epoch {step + 1} Regularization ---")
                
                # Projection phase at the end of each epoch
                self.project()
                
                # 4. Save Intermediate Pool
                try:
                    pool_path = os.path.join(intermediate_dir, f"pool_epoch_{step + 1}_reg.jsonl")
                    self.experience_pool.save(pool_path)
                except Exception as e:
                    logging.error(f"Failed to save intermediate pool: {e}")
            else:
                 logging.info(f"--- Epoch {step + 1} Regularization Skipped (Disabled) ---")
        logging.info("Training complete.")

    def _worker_task(self, worker_id: int):
        """
        Worker thread that pulls problems from the pool and processes them.
        """
        logging.info(f"Worker {worker_id} started.")
        while not self.problem_pool.empty():
            try:
                # Non-blocking get? No, use blocking with timeout to safely exit if empty
                item = self.problem_pool.get(timeout=1)
            except queue.Empty:
                break
                
            try:
                # Process the item
                # If processing returns False (due to lock), put back in queue
                success = self._process_single_item_stgd(item)
                if not success:
                    # Conflict encountered, put back to retry later
                    # Add a small random jitter to avoid hot loop collision
                    time.sleep(random.uniform(0.1, 0.5))
                    self.problem_pool.put(item)
                
                self.problem_pool.task_done()
            except Exception as e:
                logging.error(f"Worker {worker_id} error processing item: {e}")
                self.problem_pool.task_done()
        
        logging.info(f"Worker {worker_id} finished.")

    def _process_single_item_stgd(self, item) -> bool:
        """
        Processes a single item through the full TSGD cycle (Retrieve -> Solve -> Update).
        Returns True if processed successfully, False if skipped due to locking conflicts.
        """
        problem = item.get("problem", "")
        item_id = item.get("item_id", "N/A")
        
        # 1. Retrieval & Lock Check
        context = self.retrieval.run(
            problem, 
            item_id=item_id, 
            subject=item.get("subject"),
            difficulty=item.get("level") or item.get("difficulty")
        )
        
        # Check locks on retrieved experiences
        acquired_exps = []
        try:
            for exp in context:
                if exp.is_write_locked():
                    # Conflict! Back off.
                    logging.debug(f"[{item_id}] Exp {exp.id} is write locked. Backing off.")
                    return False
                
                # Attempt to acquire read lock
                if exp.acquire_read_lock():
                    acquired_exps.append(exp)
                else:
                    # Failed to acquire (maybe became write locked in between)
                    logging.debug(f"[{item_id}] Failed to acquire read lock for {exp.id}. Backing off.")
                    return False
            
            # 2. Forward Pass (Solver)
            solve_result = self.solver.solve(problem, context, item_id=item_id)
            pred_answer = extract_answer(solve_result["prediction"])
            gt_answer = item.get("ground_truth", "")
            is_correct = math_equal(pred_answer, gt_answer)
            
            # Update Usage Stats (Thread-safe atomic increment usually safe in Python GIL, but nice to be explicit)
            # We hold read locks, so updating usage count is safe if usage_count is atomic or protected?
            # Usage count is not critical, so standard += 1 is fine.
            for exp in acquired_exps:
                exp.usage_count += 1
                if is_correct:
                    exp.success_count += 1

            # If correct, we are done.
            if is_correct:
                # Add prefix to CORRECT log
                logging.info(f"[{item_id}] CORRECT")
                return True
                
            logging.info(f"[{item_id}] WRONG. Proceeding to Update.")
            
            # 3. Backward Pass (Optimizer)
            # We are inside a thread, so we generate gradient for THIS item only.
            error_context = {
                "item": item,
                "solve_result": solve_result,
                "context": context
            }
            
            # We wrap single error in list for the optimizer agent interface
            candidate_actions = self._backward_pass([error_context])
            
            # 4. Update Pass
            if candidate_actions:
                self._update_pass_single_thread(candidate_actions, [error_context])

            return True

        finally:
            # Release all read locks
            for exp in acquired_exps:
                exp.release_read_lock()

    def _update_pass_single_thread(self, actions: List[Dict], triggering_errors: List[Dict]):
        """
        Applies updates in the worker thread context.
        Handles Write Locking for EDIT/DELETE actions.
        """
        for action in actions:
            act_type = action.get("action", "").upper()
            
            if act_type == "ADD":
                # ADD is safe, just appending to list (assuming ExperiencePool.add is thread-safe enough or uses list append)
                # We reuse the direct application logic for speed
                self._apply_action_directly(action, triggering_errors)
                
            elif act_type in ["EDIT", "DELETE"]:
                # Needs Write Lock
                target_id = action.get("target_id") or action.get("id")
                target_exp = self.experience_pool.get_by_id(target_id)
                
                if not target_exp:
                    continue
                    
                # We currently hold a Read Lock on this experience (it was in context).
                # We need to Release Read Lock -> Acquire Write Lock.
                # Note: `_process_single_item_stgd` will release all read locks in `finally`.
                # BUT, to perform the update *now*, we must drop our read lock temporarily.
                # However, `_process_single_item_stgd` manages the read locks list `acquired_exps`.
                # If we release it here, we should remove it from `acquired_exps` so it doesn't get double released.
                
                # Tricky: We can't modify `acquired_exps` of the caller easily without return values.
                # Better approach: Just try to acquire Write Lock? 
                # We can't acquire Write Lock if WE hold the Read Lock. Deadlock.
                
                # Correct Logic:
                # 1. Temporarily release Read Lock on target_exp.
                target_exp.release_read_lock()
                
                try:
                    # 2. Try to Acquire Write Lock
                    # Give it a short timeout. If can't acquire, skip update.
                    if target_exp.acquire_write_lock(timeout=2.0):
                        try:
                            # Apply Update
                            if self.enable_dual_verification:
                                # Verification might be slow and involve LLM calls.
                                # Inside a worker thread, this is okay.
                                self._apply_action_with_verification(action, triggering_errors)
                            else:
                                self._apply_action_directly(action, triggering_errors)
                        finally:
                            target_exp.release_write_lock()
                    else:
                        logging.warning(f"[{triggering_errors[0]['item'].get('item_id')}] Failed to acquire write lock for {target_id}. Skipping update.")
                finally:
                    # 3. Re-acquire Read Lock to satisfy the caller's `finally` block
                    # If we can't re-acquire (e.g. someone else write locked it), 
                    # the caller's release will fail or decrement count incorrectly if we don't handle it.
                    # Actually, `release_read_lock` checks `_readers_count > 0`.
                    # If we successfully released it, count went down.
                    # We should probably NOT re-acquire, but rather tell caller we released it.
                    # But `_process_single_item_stgd` iterates `acquired_exps`.
                    
                    # Hack: Re-acquire to keep state consistent for caller.
                    # If we can't re-acquire (blocked), we might just have to wait.
                    # But we just released the write lock, so we should be able to read lock unless another writer jumped in.
                    target_exp.acquire_read_lock()

    def _train_step(self, batch_data: List[Dict]):
        """Legacy batch method - Deprecated/Unused in threaded mode"""
        pass


    def _ensure_val_embeddings(self):
        """Computes embeddings for the validation set, preferably in parallel."""
        if self.val_embeddings is not None or not self.val_data:
            return

        # Try to load from local file first
        val_path = self.config.get("val_data")
        npz_path = None
        if val_path and isinstance(val_path, str) and os.path.exists(val_path):
             npz_path = os.path.splitext(val_path)[0] + "_idx.npz"
             if not os.path.exists(npz_path):
                 pass

             if os.path.exists(npz_path):
                 try:
                     data = np.load(npz_path)
                     if "embeddings" in data:
                        loaded_embeddings = data["embeddings"]
                     elif "arr_0" in data:
                        loaded_embeddings = data["arr_0"]
                     else:
                        loaded_embeddings = None
                     
                     if loaded_embeddings is not None and len(loaded_embeddings) >= len(self.val_data):
                         # Assuming val_data is a slice from the beginning, we can slice the embeddings
                         self.val_embeddings = loaded_embeddings[:len(self.val_data)]
                         logging.info(f"Loaded validation embeddings from {npz_path} (Subset: {len(self.val_embeddings)}/{len(loaded_embeddings)})")
                         return
                     else:
                         if loaded_embeddings is not None:
                            logging.warning(f"Existing embeddings at {npz_path} have fewer items ({len(loaded_embeddings)}) than current validation set ({len(self.val_data)}). Recomputing...")
                 except Exception as e:
                     logging.warning(f"Failed to load existing embeddings from {npz_path}: {e}")

        logging.info(f"Computing embeddings for {len(self.val_data)} validation samples (Parallel)...")
        problems = [item.get("problem", "") for item in self.val_data]
        
        from concurrent.futures import ThreadPoolExecutor
        
        def embed_chunk(chunk):
            retries = 3
            for attempt in range(retries):
                try:
                    return self.experience_pool.embeddings_model.embed_documents(chunk)
                except Exception as e:
                    logging.warning(f"Embedding chunk failed (Attempt {attempt+1}/{retries}): {e}")
                    time.sleep(2 * (attempt + 1))
            logging.error(f"Embedding chunk failed after {retries} attempts.")
            return []

        # Chunk size for API calls
        chunk_size = 20 
        chunks = [problems[i:i + chunk_size] for i in range(0, len(problems), chunk_size)]
        
        all_embeddings = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(embed_chunk, chunks)
            for res in results:
                all_embeddings.extend(res)
                
        if len(all_embeddings) == len(problems):
            self.val_embeddings = np.array(all_embeddings)
            logging.info(f"Computed embeddings for {len(self.val_embeddings)} validation samples.")
            
            # Save to file if path is available
            if npz_path:
                try:
                    np.savez_compressed(npz_path, embeddings=self.val_embeddings)
                    logging.info(f"Saved validation embeddings to {npz_path}")
                except Exception as e:
                    logging.error(f"Failed to save embeddings to {npz_path}: {e}")
        else:
            logging.error(f"Failed to compute all embeddings. Expected {len(problems)}, got {len(all_embeddings)}")
            self.val_embeddings = None

    def _get_validation_subset(self, query_content: str, subject: str = None, difficulty: str = None, k: int = 5) -> List[Dict]:
        """
        Retrieves k most similar problems from validation set using Hybrid Retrieval.
        Strategy:
        1. Filter by Subject (if provided).
        2. Filter by Difficulty (if provided).
        3. Rank remaining candidates by Embedding Similarity.
        """
        if not self.val_data:
            return []
            
        self._ensure_val_embeddings()
        
        # 1. Filter candidates
        candidates = []
        candidate_indices = []
        
        for i, item in enumerate(self.val_data):
            # Subject Filter
            if subject and item.get("subject") and item.get("subject") != subject:
                continue
            
            # Difficulty Filter (Loose matching, e.g. allow +/- 1 level if we had levels)
            # Use 'level' as per dataset schema, fallback to 'difficulty' if missing
            item_level = item.get("level") or item.get("difficulty")
            if difficulty and item_level and str(item_level) != str(difficulty):
                # Optional: Relax this if too strict
                pass
                
            candidates.append(item)
            candidate_indices.append(i)
            
        # Fallback if filtering removed everyone
        if len(candidates) < k:
            logging.warning(f"Validation subset filtering (Sub={subject}, Diff={difficulty}) left only {len(candidates)} items. Using all val data.")
            candidates = self.val_data
            candidate_indices = list(range(len(self.val_data)))
            
        actual_k = min(k, len(candidates))
        
        if self.val_embeddings is None:
             # Fallback to random sampling if embedding fails
            return random.sample(candidates, actual_k)

        try:
            query_emb = self.experience_pool.embeddings_model.embed_query(query_content)
            query_vec = np.array(query_emb).reshape(1, -1)
            
            # Get embeddings for candidates only
            candidate_embeddings = self.val_embeddings[candidate_indices]
            
            # Compute Cosine Similarity
            sims = cosine_similarity(query_vec, candidate_embeddings)[0]
            
            # Get Top K
            top_k_local_indices = sims.argsort()[-actual_k:][::-1]
            
            return [candidates[i] for i in top_k_local_indices]
        except Exception as e:
            logging.error(f"Similarity search failed: {e}")
            return random.sample(candidates, actual_k)

    def _train_step(self, batch_data: List[Dict]):
        """
        Runs one step of TSGD on a batch of data.
        1. Forward: Mine errors.
        2. Backward: Aggregate errors and calculate textual gradients.
        3. Update: Dual Verification (Local + Global).
        4. Regularize: Merge and cluster experiences.
        """
        # 1. Forward Pass (Mining)
        results, errors = self._forward_pass(batch_data)
        
        if not errors:
            logging.info("No errors found in this batch. Skipping backward pass.")
            return

        # 2. Backward Pass (Gradient Estimation)
        candidate_actions = self._backward_pass(errors)
        
        if not candidate_actions:
            logging.info("No actions to perform (Optimizer returned empty). Skipping update.")
        else:
            # 3. Update Step (Dual Verification)
            self._update_pass(candidate_actions, errors)

        # 4. Regularize
        self._regularize_pass()

    def _forward_pass(self, batch_data: List[Dict]):
        logging.info(f"--- 1. Forward Pass (Mining) ---")
        logging.info(f"--- Starting Forward Pass (Batch Size: {len(batch_data)}) ---")
        errors = []
        results = []
        
        # Use max_workers from config or default to batch size (capped)
        max_workers = min(len(batch_data), self.config.get("max_workers", 5))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(self._process_single_item, item): item for item in batch_data}
            for future in as_completed(future_to_item):
                res = future.result()
                if res:
                    results.append(res)
        
        # Process results in main thread (Update stats, collect errors)
        correct_count = 0
        for res in results:
            item_id = res["item_id"]
            solve_result = res["solve_result"]
            
            # Update Usage Stats
            used_exp_ids = solve_result.get("used_exp_ids", [])
            if used_exp_ids:
                # Add prefix to Used Experiences log
                logging.info(f"[{item_id}] Used Experiences: {used_exp_ids}")
            
            for eid in used_exp_ids:
                exp = self.experience_pool.get_by_id(eid)
                if exp:
                    exp.usage_count += 1
            
            if res["is_correct"]:
                # Add prefix to Result CORRECT log
                logging.info(f"[{item_id}] Result: CORRECT")
                correct_count += 1
                # Correct! Increment success_count
                for eid in used_exp_ids:
                    exp = self.experience_pool.get_by_id(eid)
                    if exp:
                        # Add prefix to Success Count log
                        logging.info(f"[{item_id}] Success Count: {exp.success_count}")
                        exp.success_count += 1
            else:
                # Add prefix to WRONG Result log
                logging.info(f"[{item_id}] Result: WRONG (Pred: {res['pred_answer']}, GT: {res['gt_answer']})")
                # Error found
                errors.append({
                    "item": res["item"],
                    "solve_result": solve_result,
                    "context": res["context"]
                })
        
        accuracy = (correct_count / len(results)) * 100 if results else 0
        logging.info(f"Forward Pass Complete. Accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")
        return results, errors

    def _process_single_item(self, item):
        problem = item.get("problem", "")
        gt_answer = item.get("ground_truth", "")
        item_id = item.get("item_id", "N/A")
        
        # Add prefix to Processing item log
        logging.info(f"Processing item [ID: {item_id}]")
        
        try:
            context = self.retrieval.run(
                problem, 
                item_id=item_id, 
                subject=item.get("subject"),
                difficulty=item.get("level") or item.get("difficulty")
            )
            if context:
                logging.info(f"[{item_id}] Retrieved {len(context)} experiences.")
                for idx, exp in enumerate(context):
                    # Add prefix to Experience debug log
                    logging.debug(f"[{item_id}] Experience {idx+1}: {json.dumps(exp.to_dict(), indent=2, ensure_ascii=False)}")
            else:
                logging.info(f"[{item_id}] No experiences retrieved.")

            solve_result = self.solver.solve(problem, context, item_id=item_id)
            
            pred_answer = extract_answer(solve_result["prediction"])
            is_correct = math_equal(pred_answer, gt_answer)
            
            return {
                "item": item,
                "solve_result": solve_result,
                "context": context,
                "is_correct": is_correct,
                "pred_answer": pred_answer,
                "gt_answer": gt_answer,
                "item_id": item_id
            }
        except Exception as e:
            logging.error(f"Error processing item {item_id}: {e}")
            return None

    def _backward_pass(self, errors):
        logging.info(f"--- 2. Backward Pass (Gradient Estimation) ---")
        logging.info(f"--- Starting Backward Pass ({len(errors)} errors) ---")
        
        # Limit errors passed to optimizer to avoid context overflow
        # If too many errors, we can sample or take the first N.
        # Here we take up to 5 random errors to get diverse gradients.
        max_opt_errors = self.config.get("max_opt_errors", 5)
        if len(errors) > max_opt_errors:
            logging.warning(f"Too many errors ({len(errors)}). Sampling {max_opt_errors} for optimizer.")
            errors_to_process = random.sample(errors, max_opt_errors)
        else:
            errors_to_process = errors
            
        # Prepare error batch for optimizer
        error_batch_for_opt = []
        max_traj_len = self.config.get("max_traj_len", 1000)
        
        for err in errors_to_process:
            item = err["item"]
            solve_result = err["solve_result"]
            used_exps = [self.experience_pool.get_by_id(eid) for eid in solve_result.get("used_exp_ids", [])]
            used_exps = [e for e in used_exps if e is not None]
            
            # Truncate trajectory if too long
            traj = solve_result["prediction"]
            if len(traj) > max_traj_len:
                traj = traj[:max_traj_len] + "...(truncated)"
                
            error_batch_for_opt.append({
                "item_id": item.get("item_id", "N/A"),
                "problem": item["problem"],
                "gt_solution": item["solution"],
                "error_cases": traj,
                "used_experiences": used_exps,
                "context": err.get("context", [])
            })
            
        optimizer_actions = self.optimizer.run(error_batch_for_opt)
        
        # Generate prefix for logging
        item_ids = sorted(list(set(str(e.get("item_id", "N/A")) for e in error_batch_for_opt)))
        if not item_ids:
            prefix = ""
        elif len(item_ids) == 1:
            prefix = f"[{item_ids[0]}]"
        else:
            prefix = f"[{','.join(item_ids[:3])}{'...' if len(item_ids)>3 else ''}]"

        logging.info(f"{prefix} Optimizer suggested {len(optimizer_actions)} actions.")
        for idx, action in enumerate(optimizer_actions):
            act_type = action.get("action", "UNKNOWN").upper()
            logging.debug(f"{prefix} Action [{act_type}] {idx+1}: {json.dumps(action, indent=2, ensure_ascii=False)}")
            
        return optimizer_actions

    def _update_pass(self, actions, errors):
        logging.info(f"--- 3. Dual Verification & Update ---")
        logging.info("--- Starting Update Step (Dual Verification) ---")
        for i, action in enumerate(actions):
            logging.info(f"Applying action {i+1}/{len(actions)}: {action.get('action')} {action.get('id', '')}")
            if self.enable_dual_verification:
                self._apply_action_with_verification(action, errors)
            else:
                self._apply_action_directly(action, errors)

    def _apply_action_directly(self, action, triggering_errors):
        """
        Applies an action directly without verification (Fast Mode).
        """
        # Generate prefix for logging
        item_ids = sorted(list(set(str(e["item"].get("item_id", "N/A")) for e in triggering_errors))) if triggering_errors else []
        if not item_ids:
            prefix = ""
        elif len(item_ids) == 1:
            prefix = f"[{item_ids[0]}]"
        else:
            prefix = f"[{','.join(item_ids[:3])}{'...' if len(item_ids)>3 else ''}]"

        act_type = action.get("action", "").upper()
        
        if act_type == "ADD":
            if not triggering_errors:
                logging.warning(f"{prefix} [Action ADD] No triggering errors context. Skipping.")
                return
                
            trigger_item = triggering_errors[0]["item"]
            pid = trigger_item.get("item_id")
            level = trigger_item.get("level") or trigger_item.get("difficulty")
            
            # Register problem for retrieval
            self.retrieval.register_problem(
                pid, 
                trigger_item.get("problem", ""), 
                trigger_item.get("subject"), 
                level
            )

            # Validate required fields
            cond = action.get("condition")
            strat = action.get("strategy")
            if not cond or not strat:
                logging.warning(f"{prefix} [Action ADD] Skipped due to missing condition/strategy. Action: {action}")
                return

            temp_exp = Experience(
                condition=cond, 
                strategy=strat,
                source_id=[pid], # Link to first for now
                created_by_agent="Pi_opt",
                subject=trigger_item.get("subject") or "General",
                level=level,
                warning=action.get("warning", "")
            )
            # Add prefix to Action ADD Creating new experience log
            logging.info(f"{prefix} [Action ADD] Creating new experience (Fast): {temp_exp.condition[:50]}...")
            self.experience_pool.add(temp_exp)
            self.retrieval.link_experience(temp_exp.id, pid)
            self._save_pool_snapshot("add")

        elif act_type == "EDIT":
            exp_id = action.get("id")
            original_exp = self.experience_pool.get_by_id(exp_id)
            if original_exp:
                logging.info(f"{prefix} [Action EDIT] Modifying experience {exp_id[:8]} (Fast)")
                
                # Update content
                new_source_ids = list(set(original_exp.source_id + [e["item"].get("item_id") for e in triggering_errors]))
                
                # Register new source problems
                for e in triggering_errors:
                    itm = e["item"]
                    self.retrieval.register_problem(
                        itm.get("item_id"),
                        itm.get("problem", ""),
                        itm.get("subject"),
                        itm.get("level") or itm.get("difficulty")
                    )

                self.experience_pool.update(
                    exp_id, 
                    condition=action.get("condition"), 
                    strategy=action.get("strategy"),
                    warning=action.get("warning", "")
                )
                original_exp.source_id = new_source_ids
                
                # Update links
                for sid in new_source_ids:
                    self.retrieval.link_experience(exp_id, sid)
                self._save_pool_snapshot("edit")
        
        elif act_type == "DELETE":
            exp_id = action.get("id")
            if not self.experience_pool.get_by_id(exp_id):
                # Add prefix to Action DELETE Failed log
                logging.warning(f"{prefix} [Action DELETE] Failed: Experience {exp_id} not found.")
                return

            # Add prefix to Action DELETE Deleting experience log
            logging.info(f"{prefix} [Action DELETE] Deleting experience {exp_id[:8]} (Fast)")
            self.experience_pool.delete(exp_id)
            self._save_pool_snapshot("delete")


    def _apply_action_with_verification(self, action, triggering_errors):
        """
        Applies an action only if it passes dual verification.
        """
        # Generate prefix for logging
        item_ids = sorted(list(set(str(e["item"].get("item_id", "N/A")) for e in triggering_errors))) if triggering_errors else []
        if not item_ids:
            prefix = ""
        elif len(item_ids) == 1:
            prefix = f"[{item_ids[0]}]"
        else:
            prefix = f"[{','.join(item_ids[:3])}{'...' if len(item_ids)>3 else ''}]"

        act_type = action.get("action", "").upper()
        
        # (a) Local Verification: Does it fix the triggering error?
        # For ADD/EDIT, we can test on one of the triggering problems
        if triggering_errors and act_type in ["ADD", "EDIT"]:
            # Check if action fixes ANY of the triggering errors
            is_fixed = False
            test_error = None
            
            for err in triggering_errors:
                item = err["item"]
                item_id = item.get("item_id", "N/A")
                
                # Use cached retrieval if possible? No, we need new context with the new/edited experience.
                # To save time, we can reuse context but append/replace the specific experience?
                # For now, let's re-run retrieval to be safe (Simulate real inference).
                
                # Apply action (Temporary)
                # ... (We need to apply before testing)
            
            # Move application logic OUT of the loop, then test loop
            
            # Temporary apply
            temp_exp = None
            original_exp = None
            old_cond, old_strat, old_source_ids = None, None, None
            
            if act_type == "ADD":
                trigger_item = triggering_errors[0]["item"]
                pid = trigger_item.get("item_id")
                level = trigger_item.get("level") or trigger_item.get("difficulty")
                
                # Register problem for retrieval
                self.retrieval.register_problem(
                    pid, 
                    trigger_item.get("problem", ""), 
                    trigger_item.get("subject"), 
                    level
                )

                # Validate required fields
                cond = action.get("condition")
                strat = action.get("strategy")
                if not cond or not strat:
                    logging.warning(f"{prefix} [Action ADD] Skipped due to missing condition/strategy. Action: {action}")
                    return

                temp_exp = Experience(
                    condition=cond, 
                    strategy=strat,
                    source_id=[pid], # Link to first for now
                    created_by_agent="Pi_opt",
                    subject=trigger_item.get("subject") or "General",
                    level=level,
                    warning=action.get("warning", "")
                )
                # Add prefix to Action ADD Creating new experience log
                logging.info(f"{prefix} [Action ADD] Creating new experience: {temp_exp.condition[:50]}...")
                self.experience_pool.add(temp_exp)
                self.retrieval.link_experience(temp_exp.id, pid)
            else: # EDIT
                exp_id = action.get("id")
                original_exp = self.experience_pool.get_by_id(exp_id)
                if original_exp:
                    old_cond, old_strat = original_exp.condition, original_exp.strategy
                    old_warning = getattr(original_exp, "warning", "")
                    old_source_ids = original_exp.source_id.copy()
                    
                    logging.info(f"{prefix} [Action EDIT] Modifying experience {exp_id[:8]}")
                    
                    # Update content
                    new_source_ids = list(set(old_source_ids + [e["item"].get("item_id") for e in triggering_errors]))
                    
                    # Register new source problems
                    for e in triggering_errors:
                        itm = e["item"]
                        self.retrieval.register_problem(
                            itm.get("item_id"),
                            itm.get("problem", ""),
                            itm.get("subject"),
                            itm.get("level") or itm.get("difficulty")
                        )

                    self.experience_pool.update(
                        exp_id, 
                        condition=action.get("condition"), 
                        strategy=action.get("strategy"),
                        warning=action.get("warning", "")
                    )
                    original_exp.source_id = new_source_ids
                    
                    # Update links
                    for sid in new_source_ids:
                        self.retrieval.link_experience(exp_id, sid)
            
            # Verify Local Loop
            if not is_fixed:
                # Add prefix to Local Verification Testing on triggering errors log
                logging.info(f"{prefix} [Local Verification] Testing on {len(triggering_errors)} triggering errors...")
            
            for i, err in enumerate(triggering_errors):
                item = err["item"]
                item_id = item.get("item_id", "N/A")
                
                # Limit local verification to first 3 to save time if batch is large
                if i >= 3: 
                    break
                    
                logging.debug(f"   [Local Verification] Testing error {i+1}: {item_id}")
                
                # Use cached context from the error, avoiding re-retrieval
                old_context = err.get("context", [])
                
                # Construct new_context based on action
                if act_type == "ADD":
                    # For ADD, we use ONLY the new experience for local verification
                    # The user logic is that previous context failed, so we rely on the new rule.
                    if temp_exp:
                        new_context = [temp_exp]
                    else:
                        new_context = []
                else: # EDIT
                    # For EDIT, explicitly replace the experience in context to ensure updated version is used
                    new_context = []
                    for e in old_context:
                        if original_exp and e.id == original_exp.id:
                            new_context.append(original_exp)
                        else:
                            new_context.append(e)
                    
                new_solve = self.solver.solve(item["problem"], new_context, item_id=item_id)
                new_ans = extract_answer(new_solve["prediction"])
                
                if math_equal(new_ans, item["ground_truth"]):
                    is_fixed = True
                    test_error = err
                    # Add prefix to Local Verification FIXED log
                    logging.info(f"{prefix} [Local Verification] FIXED on item {item_id}")
                    break # Success!
                
                # Add prefix to Local Verification FAILED log
                if not is_fixed:
                    # Add prefix to Local Verification FAILED log
                    logging.info(f"{prefix} [Local Verification] FAILED (None of the tested errors were fixed)")
            
            # (b) Global/Variance Reduction: Check on a mini-val set
            is_globally_valid = True
            if is_fixed and self.val_data:
                # Define query_content for validation subset retrieval
                # Use the problem text of the fixed item as the query to find similar validation problems
                query_content = test_error["item"].get("problem", "")

                # Use the item that was fixed for context
                item = test_error["item"]
                item_id = item.get("item_id", "N/A")
                    
                mini_val_set = self._get_validation_subset(
                    query_content, 
                    subject=item.get("subject"), # Pass subject from current item
                    difficulty=item.get("difficulty"), # Pass difficulty
                    k=5
                )
                
                # Score with New Experience (already applied)
                score_new = self.evaluate(mini_val_set)
                
                # Revert to measure Score Old
                if act_type == "ADD":
                    self.experience_pool.delete(temp_exp.id)
                elif original_exp:
                    # In _apply_action_with_verification, we've already applied the change.
                    # So to measure old score, we must revert it.
                    self.experience_pool.update(
                        original_exp.id, 
                        condition=old_cond, 
                        strategy=old_strat,
                        warning=old_warning
                    )
                    original_exp.source_id = old_source_ids
                
                # Score with Old Experience
                score_old = self.evaluate(mini_val_set)
                
                # Add prefix to Global Verification logs
                logging.info(f"{prefix} [Global Verification] Score New: {score_new:.2f}")
                logging.info(f"{prefix} [Global Verification] Score Old: {score_old:.2f}")
                
                # Calculate Delta
                delta = score_new - score_old
                logging.info(f"{prefix} [Global Verification] Score Old: {score_old:.2f}, Score New: {score_new:.2f}, Delta: {delta:.2f}")
                
                # --- Dynamic Batch Adjustment ---
                is_extended_and_applied = False
                if abs(delta) < 0.05:
                    # Add prefix to Global Verification weak signal log
                    logging.info(f"{prefix} [Global Verification] Signal weak (|Delta| < 0.05). Extending validation set...")
                    # Fetch more samples (k=10 total)
                    full_val_set = self._get_validation_subset(
                        query_content, 
                        subject=item.get("subject"), 
                        difficulty=item.get("level") or item.get("difficulty"),
                        k=10
                    )
                    
                    # Identify new items
                    existing_ids = set(x.get("item_id") for x in mini_val_set)
                    new_items = [x for x in full_val_set if x.get("item_id") not in existing_ids]
                    
                    if new_items:
                        # Add prefix to Global Verification Evaluating log
                        logging.info(f"{prefix} [Global Verification] Evaluating on {len(new_items)} additional items...")
                        
                        # 1. Evaluate Old State (Current State) on New Items
                        score_old_2 = self.evaluate(new_items)
                        
                        # 2. Re-apply Action (Switch to New State)
                        if act_type == "ADD":
                            self.experience_pool.add(temp_exp)
                        elif original_exp:
                            # In _apply_action_with_verification, we've already applied the change.
                            # So to measure old score, we must revert it.
                            self.experience_pool.update(
                                original_exp.id, 
                                condition=action.get("condition"), 
                                strategy=action.get("strategy"),
                                warning=f"{prefix} {action.get("warning", "")}"
                            )
                            original_exp.source_id = list(set(old_source_ids + [item_id]))
                        
                        is_extended_and_applied = True
                        
                        # 3. Evaluate New State on New Items
                        score_new_2 = self.evaluate(new_items)
                        
                        # 4. Update Scores (Weighted Average)
                        n1 = len(mini_val_set)
                        n2 = len(new_items)
                        score_old = (score_old * n1 + score_old_2 * n2) / (n1 + n2)
                        score_new = (score_new * n1 + score_new_2 * n2) / (n1 + n2)
                        delta = score_new - score_old
                        # Add prefix to Global Verification Updated Scores log
                        logging.info(f"{prefix} [Global Verification] Updated Scores (N={n1+n2}): Old: {score_old:.2f}, New: {score_new:.2f}, Delta: {delta:.2f}")

                # Add prefix to Global Verification PASSED log
                if delta >= -Config.ACCEPTANCE_THRESHOLD:
                    logging.info(f"{prefix} [Global Verification] PASSED (Delta {delta:.2f} >= -{Config.ACCEPTANCE_THRESHOLD})")
                    # Re-apply if not already applied
                    if not is_extended_and_applied:
                        # Add prefix to Global Verification PASSED log
                        logging.info(f"{prefix} [Global Verification] PASSED (Delta {delta:.2f} >= -{Config.ACCEPTANCE_THRESHOLD})")
                        
                        # Re-apply if not already applied
                        if act_type == "ADD":
                            self.experience_pool.add(temp_exp)
                        elif original_exp:
                            self.experience_pool.update(
                                original_exp.id, 
                                condition=action.get("condition"), 
                                strategy=action.get("strategy"),
                                warning=action.get("warning", "")
                            )
                            original_exp.source_id = list(set(old_source_ids + [item_id]))
                else:
                    # Add prefix to Global Verification FAILED log
                    logging.info(f"{prefix} [Global Verification] FAILED (Delta {delta:.2f} < -{Config.ACCEPTANCE_THRESHOLD})")
                    is_globally_valid = False
                    # Revert if we applied it during extension
                    if is_extended_and_applied:
                        if act_type == "ADD":
                            self.experience_pool.delete(temp_exp.id)
                        elif original_exp:
                            self.experience_pool.update(original_exp.id, condition=old_cond, strategy=old_strat)
                            original_exp.source_id = old_source_ids
            
            if is_fixed and is_globally_valid:
                # Add prefix to Action ACCEPTED log
                logging.info(f"{prefix} [Action ACCEPTED] {act_type} confirmed.")
                self._save_pool_snapshot(act_type.lower())
            else:
                logging.info(f"{prefix} [Action REJECTED] {act_type} reverted.")
                # Revert logic for failed local verification is already handled if we didn't re-apply
                if not is_fixed:
                     if act_type == "ADD":
                        self.experience_pool.delete(temp_exp.id)
                     elif original_exp:
                        self.experience_pool.update(
                            original_exp.id, 
                            condition=old_cond, 
                            strategy=old_strat, 
                            warning=old_warning
                        )
                        original_exp.source_id = old_source_ids
        
        elif act_type == "DELETE":
            exp_id = action.get("id")
            if not self.experience_pool.get_by_id(exp_id):
                logging.warning(f"[Action DELETE] Failed: Experience {exp_id} not found.")
                return

            # Add prefix to Action DELETE log
            logging.info(f"{prefix} [Action DELETE] Attempting to delete experience {exp_id[:8]}...")
            
            # 1. Temporary Apply (Delete)
            self.experience_pool.delete(exp_id)
            
            # 2. Local Verification: Does removing it fix the error?
            is_fixed = False
            target_error = None
            
            # Try to find an error that actually used this experience
            if triggering_errors:
                for err in triggering_errors:
                    used_ids = err["solve_result"].get("used_exp_ids", [])
                    if exp_id in used_ids:
                        target_error = err
                        break
                if not target_error:
                    target_error = triggering_errors[0] # Fallback
            
            if target_error:
                item = target_error["item"]
                item_id = item.get("item_id", "N/A")
                logging.info(f"   [Local Verification] Testing if DELETE fixes {item_id}...")
                
                # Use cached context from the error, avoiding re-retrieval
                old_context = target_error.get("context", [])
                
                # Filter out the deleted experience
                new_context = [e for e in old_context if e.id != exp_id]
                
                new_solve = self.solver.solve(item["problem"], new_context, item_id=item_id)
                new_ans = extract_answer(new_solve["prediction"])
                
                is_fixed = math_equal(new_ans, item["ground_truth"])
                # Add prefix to Local Verification Result log
                logging.info(f"{prefix} [Local Verification] Result: {'FIXED' if is_fixed else 'FAILED'}")
            else:
                # Add prefix to Local Verification No triggering error context log
                logging.warning(f"{prefix} [Local Verification] No triggering error context found. Skipping local check.")
                is_fixed = True # Assume valid if we can't test locally (risky but allows optimizer freedom)
                # Add prefix to Local Verification Assume Valid log
                logging.info(f"{prefix} [Local Verification] Assume Valid (No local test context)")
            

            # 3. Global Verification
            is_globally_valid = True
            if is_fixed and self.val_data:
                # Add prefix to Global Verification Running log
                logging.info(f"{prefix} [Global Verification] Running on similarity-based subset (size: 5)...")
                
                # Subset based on the problem text of the triggering error
                # This helps find validation problems structurally similar to the one where the experience failed
                query_content = target_error["item"].get("problem", "") if target_error else original_exp.condition
                mini_val_set = self._get_validation_subset(
                    f"{prefix} {query_content}", 
                    subject=original_exp.subject, # Use subject from experience
                    k=5
                )
                
                # Score with DELETE applied (New State)
                score_new = self.evaluate(mini_val_set)
                
                # Revert (Add back) to measure Old State
                self.experience_pool.add(original_exp)
                score_old = self.evaluate(mini_val_set)
                
                # Calculate Delta
                delta = score_new - score_old
                # Add prefix to Global Verification Score log
                logging.info(f"{prefix} [Global Verification] Score Old: {score_old:.2f}, Score New: {score_new:.2f}, Delta: {delta:.2f}")
                
                # --- Dynamic Batch Adjustment ---
                is_extended_and_applied = False
                if abs(delta) < 0.05:
                    logging.info(f"{prefix} [Global Verification] Signal weak (|Delta| < 0.05). Extending validation set...")
                    full_val_set = self._get_validation_subset(
                        query_content, 
                        subject=original_exp.subject, 
                        k=10
                    )
                    existing_ids = set(x.get("item_id") for x in mini_val_set)
                    new_items = [x for x in full_val_set if x.get("item_id") not in existing_ids]
                    
                    if new_items:
                        # Add prefix to Global Verification Evaluating on additional items log
                        logging.info(f"{prefix} [Global Verification] Evaluating on {len(new_items)} additional items...")
                        
                        # 1. Evaluate Old State (Current State: Reverted/Added back) on New Items
                        score_old_2 = self.evaluate(new_items)
                        
                        # 2. Re-apply Action (DELETE) (Switch to New State)
                        self.experience_pool.delete(exp_id)
                        is_extended_and_applied = True
                        # Add prefix to Global Verification Extended Validation Set log
                        logging.info(f"{prefix} [Global Verification] Extended Validation Set (N={len(full_val_set)}).")

                        # 3. Evaluate New State on New Items
                        score_new_2 = self.evaluate(new_items)
                        
                        # 4. Update Scores
                        n1 = len(mini_val_set)
                        n2 = len(new_items)
                        score_old = (score_old * n1 + score_old_2 * n2) / (n1 + n2)
                        score_new = (score_new * n1 + score_new_2 * n2) / (n1 + n2)
                        delta = score_new - score_old
                        # Add prefix to Global Verification Updated Scores log
                        logging.info(f"{prefix} [Global Verification] Updated Scores (N={n1+n2}): Old: {score_old:.2f}, New: {score_new:.2f}, Delta: {delta:.2f}")
                
                # Add prefix to Global Verification PASSED log
                if delta >= -Config.ACCEPTANCE_THRESHOLD:
                    logging.info(f"{prefix} [Global Verification] PASSED (Delta {delta:.2f} >= -{Config.ACCEPTANCE_THRESHOLD})")
                    # Re-apply DELETE if not already applied
                    if not is_extended_and_applied:
                        self.experience_pool.delete(exp_id)
                else:
                    # Add prefix to Global Verification FAILED log
                    logging.info(f"{prefix} [Global Verification] FAILED (Delta {delta:.2f} < -{Config.ACCEPTANCE_THRESHOLD})")
                    is_globally_valid = False
                    # Revert (Add back) if we applied it during extension
                    if is_extended_and_applied:
                        self.experience_pool.add(original_exp)
            
            elif not is_fixed:
                # If local verification failed, we must revert
                self.experience_pool.add(original_exp)

            if is_fixed and is_globally_valid:
                # Add prefix to Action ACCEPTED log
                logging.info(f"{prefix} [Action ACCEPTED] {act_type} confirmed.")
                self._save_pool_snapshot(act_type.lower())
            else:
                logging.info(f"{prefix} [Action REJECTED] {act_type} reverted.")

    def evaluate(self, data: List[Dict]) -> float:
        """Evaluates current pool on a dataset (Parallel)."""
        if not data:
            return 0.0
            
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        correct = 0
        total = len(data)
        
        # Use a local lock for thread-safe counter increment
        # LOG_LOCK is imported globally for logging
        count_lock = threading.Lock()
        
        def process_item(item):
            # item_id is crucial for identifying logs
            item_id = item.get("item_id", "N/A")
            problem = item.get("problem", "")
            gt = item.get("ground_truth", "")
            
            try:
                # 1. Retrieval
                context = self.retrieval.run(problem, item_id=item_id)
                
                # 2. Solver
                res = self.solver.solve(problem, context, item_id=item_id)
                
                # 3. Grading
                pred = extract_answer(res["prediction"])
                is_correct = math_equal(pred, gt)
                
                return is_correct
            except Exception as e:
                with LOG_LOCK:
                    logging.error(f"[{item_id}] Evaluation failed: {e}")
                return False

        # Use ThreadPoolExecutor for I/O bound tasks (LLM API calls)
        # Max workers can be tuned based on API rate limits
        max_workers = self.config.get("eval_max_workers", 5)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {executor.submit(process_item, item): item for item in data}
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_item), total=total, desc="Evaluating (Parallel)"):
                try:
                    is_correct = future.result()
                    if is_correct:
                        correct += 1
                except Exception as e:
                    logging.error(f"Evaluation task exception: {e}")
                    
        return correct / total

    def project(self):
        """
        Projection phase: Regularize and stabilize the experience pool.
        """
        logging.info("Starting Projection phase (Regularization)...")
        
        # Use configurable limits from self.config (CLI)
        total_limit = self.config.get("total_limit", Config.TOTAL_LIMIT)
        subject_limit = self.config.get("subject_limit", Config.SUBJECT_LIMIT)
        # Use exp_max_tokens for experience length, fallback to max_tokens or default
        max_tokens = self.config.get("exp_max_tokens", self.config.get("max_tokens", Config.MAX_TOKENS))
        
        self.regularizer.run(
            total_limit=total_limit,
            subject_limit=subject_limit,
            max_tokens=max_tokens,
            mode="problem"  # Explicitly enforce problem-based clustering
        )
        self._save_pool_snapshot("regularize")
        logging.info(f"Projection complete. Pool size: {len(self.experience_pool.experiences)}")

import argparse
import sys
from src.tools.utils import (
    DataLoader, 
    setup_logging, 
    setup_phoenix, 
    Config,
    add_common_arguments
)

def main():
    parser = argparse.ArgumentParser(description="TSGD Trainer (TSGD)")
    parser = add_common_arguments(parser)
    
    # Trainer Specific Arguments
    trainer_group = parser.add_argument_group("Trainer Specific (TSGD)")
    trainer_group.add_argument("--train_data", type=str, help="Path to training data file")
    trainer_group.add_argument("--val_data", type=str, help="Path to validation data file")
    trainer_group.add_argument("--test_data", type=str, help="Path to test data file")
    trainer_group.add_argument("--seed_samples", type=int, default=10, help="Number of samples for initial seeding")
    trainer_group.add_argument("--train_samples", type=int, default=20, help="Number of samples for each training epoch")
    trainer_group.add_argument("--val_samples", type=int, default=20, help="Number of samples for validation")
    trainer_group.add_argument("--test_samples", type=int, default=20, help="Number of samples for testing")
    trainer_group.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    trainer_group.add_argument("--enable_dual_verification", action="store_true",  default=False, help="Enable Dual Verification (perform verification)")
    trainer_group.add_argument("--enable_regularizer", action="store_true", default=False, help="Enable regularizer (perform regularization/projection)")
    trainer_group.add_argument("--regularization_epsilon", type=float, default=1e-6, help="Epsilon for utility stability")
    trainer_group.add_argument("--embedding_path", type=str, default="", help="Path to the problem embedding matrix (.npz)")
    trainer_group.add_argument("--log_file", type=str, help="Path to custom log file")
    
    args = parser.parse_args()

    # 1. Setup Environment & Global Config
    config_dict = vars(args)
    config_dict["enable_dual_verification"] = args.enable_dual_verification
    config_dict["enable_regularizer"] = args.enable_regularizer
    Config.update(config_dict)
    
    # 2. Setup Logging & Tracing
    setup_logging(log_file=args.log_file)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_phoenix(project_name=args.project_name or "explearn_trainer")
    
    # 2. Load Data
    logging.info(f"Loading dataset: {args.dataset_name}")
    try:
        # Load from explicit paths if provided, else fallback to dataset_name and split
        train_path = args.train_data if args.train_data and os.path.exists(args.train_data) else args.dataset_name
        val_path = args.val_data if args.val_data and os.path.exists(args.val_data) else args.dataset_name
        
        logging.info(f"Loading train data from: {train_path}")
        full_train_data = DataLoader.load_dataset(train_path, split="train")
        
        logging.info(f"Loading val data from: {val_path}")
        full_val_data = DataLoader.load_dataset(val_path, split="validation") # or validation if available

        test_path = args.test_data if args.test_data and os.path.exists(args.test_data) else None
        full_test_data = []
        if test_path:
            logging.info(f"Loading test data from: {test_path}")
            full_test_data = DataLoader.load_dataset(test_path, split="test")
        
        # Debugging: Log the first item's ID from each set to verify
        if full_train_data:
            logging.info(f"First train item ID: {full_train_data[0].get('item_id', 'N/A')}")
        if full_val_data:
            logging.info(f"First val item ID: {full_val_data[0].get('item_id', 'N/A')}")
        if full_test_data:
            logging.info(f"First test item ID: {full_test_data[0].get('item_id', 'N/A')}")
            
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        sys.exit(1)
        
    # Split/Sample Data
    seed_data = full_train_data[:args.seed_samples]
    train_data = full_train_data[args.seed_samples:args.seed_samples + args.train_samples]
    # Use full val/test data as requested by user
    val_data = full_val_data
    test_data = full_test_data if full_test_data else []
    
    logging.info(f"Data Split: Seed={len(seed_data)}, Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # 3. Initialize Components
    logging.info("Initializing Agents and Pool...")
    
    # Update config
    config_dict = vars(args)
    config_dict["test_data"] = test_data
    config_dict["max_workers"] = config_dict.get("max_workers", 10)

    # Prepare problem IDs for Regularizer if embedding path is present
    # This allows Regularizer to load embeddings without a separate meta file
    if args.embedding_path and full_train_data:
        problem_ids = [str(item.get("item_id", "")) for item in full_train_data]
        config_dict["problem_ids"] = problem_ids
        config_dict["problem_embedding_path"] = args.embedding_path
        logging.info(f"Configured Regularizer with {len(problem_ids)} problem IDs from training data.")

    # Try to load existing pool or create new
    pool = ExperiencePool(
        retrieval_top_k=args.retrieval_top_k,
        similarity_threshold=args.similarity_threshold,
        max_pool_size=args.total_limit
    )
    
    solver = SolverAgent(config=config_dict)
    optimizer = OptimizerAgent(config=config_dict)
    regularizer = RegularizerAgent(experience_pool=pool, config=config_dict)
    retrieval = RetrievalAgent(experience_pool=pool, config=config_dict)
    initializer = InitAgent(config=config_dict)

    if args.experience_dir and os.path.exists(args.experience_dir):
        try:
            # retrieval.load_data will also call pool.load() internally
            retrieval.load_data(embedding_path=args.embedding_path, input_dir=args.experience_dir)
            logging.info(f"Loaded existing pool and metadata from {args.experience_dir}.")
        except Exception as e:
            logging.warning(f"Failed to load existing data from {args.experience_dir}: {e}. Starting fresh.")
    
    # 4. Initialize Trainer
    trainer = Trainer(
        experience_pool=pool,
        solver=solver,
        optimizer=optimizer,
        regularizer=regularizer,
        retrieval=retrieval,
        initializer=initializer,
        val_data=val_data,
        config=config_dict
    )
    
    # 5. Run Workflow
    
    # Step A: Seeding
    if seed_data:
        trainer.seed(seed_data)
        if args.experience_dir:
            os.makedirs(args.experience_dir, exist_ok=True)
            seed_pool_path = os.path.join(args.experience_dir, "pool_seed.jsonl")
            pool.save(seed_pool_path)
            logging.info(f"Seed experiences saved to {seed_pool_path}")
        
    # Step B: Training
    # Trainer handles the loop internally based on config["epochs"]
    logging.info(f"Starting training for {args.epochs} epochs with {config_dict['max_workers']} workers")
    if train_data:
        trainer.train(train_data)
        
    # 6. Save Result
        
    # 6. Save Result
    if args.experience_dir:
        os.makedirs(args.experience_dir, exist_ok=True)
        pool.save(args.experience_dir)
        retrieval.save(args.experience_dir)
        logging.info(f"Final experience pool and metadata saved to {args.experience_dir}")
    else:
        logging.warning("No experience_dir provided. Results not saved.")

if __name__ == "__main__":
    main()
