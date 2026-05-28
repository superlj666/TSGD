import os
import json
import logging
import argparse
import sys
import concurrent.futures
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.tools.utils import (
    LOG_LOCK, 
    Config, 
    DataLoader, 
    setup_logging, 
    setup_phoenix, 
    get_chat_model,
    add_common_arguments
)
from src.agents.solver import SolverAgent
from src.agents.retriever import RetrievalAgent
from src.agents.evaluator import EvaluatorAgent, calculate_pass_at_k
from src.core.experience_pool import ExperiencePool

class InferenceEngine:
    """
    Core Inference logic: Orchestrates retrieval, solving, and evaluation for a given problem.
    Handles parallel execution over datasets and reports metrics.
    """
    def __init__(self, experience_pool: ExperiencePool, config: Dict[str, Any] = None):
        self.config = config or {}
        self.experience_pool = experience_pool
        self.solver = SolverAgent(config=self.config)
        
        # Retrieval Agent configuration
        self.retriever = RetrievalAgent(self.experience_pool, config=self.config)
        
        self.evaluator = EvaluatorAgent(config=self.config)
        
        self.debug = self.config.get("debug", False)
        self.max_workers = self.config.get("max_workers", Config.MAX_WORKERS)
        self.k = self.config.get("k", Config.K)
        self.retrieval_mode = self.config.get("retrieval_mode", "problem")
        
        # Oracle Retrieval (Meta/SourceID) setup
        self.meta_path = self.config.get("meta_path")
        self.meta_data = None
        self.source_map = {}
        
        if self.meta_path:
            if os.path.exists(self.meta_path):
                logging.info(f"Loading question metadata from {self.meta_path}...")
                try:
                    with open(self.meta_path, 'r', encoding='utf-8') as f:
                        self.meta_data = json.load(f)
                    logging.info(f"Loaded {len(self.meta_data)} entries from meta file.")
                except Exception as e:
                    logging.error(f"Failed to load meta file: {e}")
            else:
                logging.warning(f"Meta path {self.meta_path} provided but file not found. Falling back to SourceID lookup.")
        
        # Build SourceID map if needed (either no meta file provided OR meta file load failed/missing)
        # Only needed if we are using the Oracle logic (implied by presence of meta_path arg in user request context, 
        # but to be safe and efficient, let's build it if meta_data is missing but we might need it? 
        # Or just build it lazily? Let's build it now if meta_data is None.)
        if self.meta_path and not self.meta_data:
             logging.info("Building SourceID -> Experience map for fallback retrieval...")
             for exp in self.experience_pool.experiences:
                 for sid in exp.source_id:
                     if sid and sid != "N/A":
                         if sid not in self.source_map:
                             self.source_map[sid] = []
                         self.source_map[sid].append(exp)
             logging.info(f"Built SourceID map with {len(self.source_map)} source keys.")

    def run(self, dataset: List[Dict]) -> List[Dict]:
        """
        Executes inference over a dataset and returns trajectories.
        """
        logging.info(f"=== Starting Inference Engine (N={len(dataset)}) ===")
        logging.info(f"Parallel Workers: {self.max_workers}, k={self.k}")

        all_trajectories = []
        results = []
        all_reasons = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.evaluate_problem, item) for item in dataset]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(dataset), desc="Inference"):
                problem_correct_count, trajectories, reasons = future.result()
                results.append(problem_correct_count)
                all_trajectories.extend(trajectories)
                all_reasons.extend(reasons)

        self._report_metrics(results, all_reasons, len(dataset))
        return all_trajectories

    def evaluate_problem(self, item: Dict) -> tuple:
        problem_correct_count = 0
        trajectories = []
        reasons = []
        all_logs = []

        for i in range(self.k):
            res = self.evaluate_sample(item, i)
            if res["is_correct"]:
                problem_correct_count += 1
            reasons.append(res["reason"])
            trajectories.append(res["trajectory"])
            all_logs.extend(res["logs"])
        
        with LOG_LOCK:
            for log_msg in all_logs:
                logging.info(log_msg)
        
        return problem_correct_count, trajectories, reasons

    def evaluate_sample(self, item: Dict, i: int) -> Dict:
        problem = item.get('problem', '')
        ground_truth = item.get('ground_truth', '')
        solution = item.get('solution', '')
        item_id = item.get('item_id', 'N/A')
        current_id = f"{item_id}_s{i}" if self.k > 1 else item_id
        
        problem_logs = []
        def log_buffer(msg):
            problem_logs.append(msg)

        try:
            # 1. Retrieve
            subject = item.get("subject") or item.get("subject")
            
            # Check for Oracle/Meta Retrieval Logic
            context = []
            if self.meta_path:
                # Priority 1: Meta File Lookup
                if self.meta_data:
                    meta_entry = self.meta_data.get(item_id)
                    if meta_entry:
                        exp_ids = meta_entry.get("experience_ids", [])
                        
                        # Apply limits if needed
                        if self.experience_pool.retrieval_top_k:
                            exp_ids = exp_ids[:self.experience_pool.retrieval_top_k]

                        for eid in exp_ids:
                            exp = self.experience_pool.registry.get(eid)
                            if exp:
                                context.append(exp)
                
                # Priority 2: SourceID Lookup (Fallback within Oracle)
                if not context and not self.meta_data:
                    context = self.source_map.get(item_id, [])
            
            # Priority 3: Default / Standard Retrieval (Fallback if Oracle found nothing or wasn't used)
            if not context:
                # Default / Standard Retrieval
                context = self.retriever.run(problem, item_id=current_id, subject=subject, mode=self.retrieval_mode)
            
            context_dicts = [e.to_dict() for e in context]
            
            # 2. Solve
            solve_result = self.solver.solve(problem, context, item_id=current_id)
            prediction = solve_result["prediction"]
            
            # 3. Assess
            is_correct, reason = self.evaluator.assess_prediction(
                prediction, ground_truth, item_id=current_id, problem=problem, logger=log_buffer
            )
            
            log_buffer(f"[{current_id}] Correct: {is_correct} | Tokens: {solve_result['input_tokens']}/{solve_result['output_tokens']} | Latency: {solve_result['latency']:.2f}s")

            return {
                "is_correct": is_correct,
                "reason": reason,
                "logs": problem_logs,
                "trajectory": {
                    "question_id": item_id,
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "solution": solution,
                    "prediction": prediction,
                    "used_exp_ids": solve_result.get("used_exp_ids", []),
                    "is_correct": is_correct,
                    "reason": reason,
                    "latency": solve_result["latency"],
                    "input_tokens": solve_result["input_tokens"],
                    "output_tokens": solve_result["output_tokens"],
                    "context": context_dicts
                }
            }
        except Exception as e:
            log_buffer(f"[{current_id}] Error in inference: {e}")
            return {
                "is_correct": False,
                "reason": "error",
                "logs": problem_logs,
                "trajectory": {
                    "question_id": item_id,
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "solution": solution,
                    "prediction": f"Error: {str(e)}",
                    "is_correct": False,
                    "reason": "error",
                    "latency": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "context": []
                }
            }

    def _report_metrics(self, results: List[int], all_reasons: List[str], total_problems: int):
        if total_problems == 0:
            logging.info("No test data found.")
            return

        mean_k = sum(results) / (total_problems * self.k)
        pass_k_sum = sum(calculate_pass_at_k(self.k, c, self.k) for c in results)
        final_pass_k = pass_k_sum / total_problems
        
        stats = {
            "correct": all_reasons.count("correct"),
            "extraction_failed": all_reasons.count("extraction_failed"),
            "mismatch": all_reasons.count("mismatch"),
            "connection_error": all_reasons.count("connection_error"),
            "error": all_reasons.count("error")
        }
        
        logging.info(f"\nFinal Results:")
        logging.info(f"Pass@{self.k}: {final_pass_k:.4f}")
        logging.info(f"Mean@{self.k}: {mean_k:.4f}")
        logging.info(f"Total Samples: {len(all_reasons)}")
        logging.info(f"Correct: {stats['correct']}")
        logging.info(f"Extraction Failed: {stats['extraction_failed']}")
        logging.info(f"Connection Error: {stats['connection_error']}")
        logging.info(f"Answer Mismatch: {stats['mismatch']}")
        logging.info(f"Other Errors: {stats['error']}")

def main():
    parser = argparse.ArgumentParser(description="TSGD Inference Runner")
    parser = add_common_arguments(parser)
    
    # Inference Specific Arguments
    inf_group = parser.add_argument_group("Inference Specific")
    inf_group.add_argument("--output_path", type=str, help="Path to save trajectories")
    inf_group.add_argument("--retrieval_mode", type=str, default="problem", choices=["problem", "condition", "problem"], help="Mode for experience retrieval")
    inf_group.add_argument("--problem_embedding_path", type=str, help="Path to problem embeddings index")
    inf_group.add_argument("--experience_embedding_path", type=str, help="Path to experience embeddings index")
    inf_group.add_argument("--meta_path", type=str, help="Path to meta file for oracle retrieval")
    inf_group.add_argument("--rerank", action="store_true", help="Enable Agentic Rerank in retrieval")
    inf_group.add_argument("--log_file", type=str, help="Path to custom log file")

    args = parser.parse_args()

    # 1. Setup Environment & Global Config
    Config.update(vars(args))
    
    # Set random seed
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 2. Setup Logging & Tracing
    dataset_name = args.dataset_name or Config.DATASET_NAME
    log_path = setup_logging(args.log_file)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Log all arguments in JSON format
    logging.info("Arguments:\n" + json.dumps(vars(args), indent=4, ensure_ascii=False))
    
    setup_phoenix(project_name=args.project_name or "explearn_inference")

    # 3. Configure output path
    logging.info(f"Experiment started.")
    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output will be saved to: {args.output_path}")

    # 4. Load Data
    if args.dataset_name:
        os.environ["DATASET_NAME"] = args.dataset_name
    
    split = "test"
    data = DataLoader.load_data(dataset_name, split=split, max_samples=args.max_samples, seed=args.seed)


    # 5. Initialize Experience Pool and Inference Engine
    experience_pool = ExperiencePool(
        max_pool_size=args.total_limit,
        retrieval_top_k=args.retrieval_top_k, 
        similarity_threshold=args.similarity_threshold
    )
    
    if args.experience_dir and os.path.exists(args.experience_dir):
        logging.info(f"Loading pre-existing experience pool from {args.experience_dir}...")
        experience_pool.load(args.experience_dir)
    elif args.experience_dir:
        logging.warning(f"Experience pool directory {args.experience_dir} specified but not found.")

    engine = InferenceEngine(
        experience_pool=experience_pool,
        config=vars(args)
    )

    # Load problem matrix for retrieval if applicable
    # We must prioritize finding the directory that contains the metadata
    meta_search_dir = args.experience_dir

    if hasattr(args, 'problem_embedding_path') and args.problem_embedding_path:
        exp_idx_path = getattr(args, 'experience_embedding_path', None)
        engine.retriever.load_data(
            embedding_path=args.problem_embedding_path, 
            input_dir=meta_search_dir,
            experience_embedding_path=exp_idx_path
        )
    elif args.experience_dir:
        # Try to infer paths
        base_dir = args.experience_dir
        potential_npz = os.path.join(base_dir, "question_indexing.npz")
        exp_idx_path = os.path.join(base_dir, "experience_idx.npz")
        
        if not os.path.exists(potential_npz):
            data_root = os.path.dirname(base_dir)
            if os.path.exists(data_root):
                dataset_prefix = os.path.basename(base_dir).split('_train_')[0]
                for f in os.listdir(data_root):
                    if f.startswith(dataset_prefix) and f.endswith("_idx.npz"):
                        potential_npz = os.path.join(data_root, f)
                        logging.info(f"Found related embedding file in data root: {potential_npz}")
                        break

        if os.path.exists(potential_npz):
            logging.info(f"Inferred problem embedding path: {potential_npz}")
            engine.retriever.load_data(
                embedding_path=potential_npz, 
                input_dir=meta_search_dir,
                experience_embedding_path=exp_idx_path
            )
        else:
            logging.error(f"Could not find problem embedding matrix (.npz) for {args.experience_path}")

    # 6. Run Inference
    logging.info("Starting inference...")
    trajectories = engine.run(data)

    # 7. Save Results
    if args.output_path:
        logging.info(f"Saving {len(trajectories)} trajectories to {args.output_path}")
        with open(args.output_path, 'w', encoding='utf-8') as f:
            for traj in trajectories:
                f.write(json.dumps(traj) + '\n')
    else:
        logging.info("Skipping saving results as no output_path was provided.")

    logging.info("Experiment completed successfully.")
    logging.info(f"Log saved to: {log_path}")
    if args.output_path:
        logging.info(f"Results saved to: {args.output_path}")
    

if __name__ == "__main__":
    main()
