import json
import logging
import time
import os
import argparse
import sys
import numpy as np
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add the project root to sys.path to allow importing from src.tools.utils etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.agents.base_agent import BaseAgent
from src.core.experience_pool import Experience, ExperiencePool
from src.tools.utils import get_embeddings_model, Config, LOG_LOCK, DataLoader, setup_logging, setup_phoenix

class InitAgent(BaseAgent):
    """
    Initializer Agent (Pi_init):
    Extracts atomic experiences from Ground Truth trajectories.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.llm = self._init_llm("initializer")
    
    def _extract_json(self, text: str) -> str:
        """
        Robustly extracts JSON from LLM response.
        """
        # 1. Try markdown blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # 2. Try to find the first [ or { and the last ] or }
        import re
        match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            text = text.strip()
            
        # 3. Fix common JSON issues from LLM (like invalid backslashes in LaTeX)
        # Only escape backslashes that are NOT part of a valid escape sequence (\", \\, \/, \b, \f, \n, \r, \t, \uXXXX)
        # But a simpler way is to just escape all backslashes that are not already escaped, 
        # or more simply, just escape them and then fix the already escaped ones.
        # Actually, for LaTeX, we mostly care about things like \theta, \(, \), \frac etc.
        # A common trick is to replace \ with \\, then replace \\\\ back to \\.
        fixed_text = text.replace('\\', '\\\\')
        # Fix already escaped sequences that are now triple-escaped or double-escaped incorrectly
        # If we had \n -> \\n, now we have \\n -> \\\\n. We want it to be \\n.
        # This is tricky. Let's try a more targeted approach for common LaTeX patterns.
        
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            # If it fails, try to escape backslashes
            # This is a common pattern in LLM-generated JSON with LaTeX
            # We want to escape backslashes that are followed by characters that don't form valid escape sequences
            # Valid escapes: [" \ / b f n r t u]
            text = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', text)
            return text

    def run(self, problem: str, solution: str, item_id: str = "N/A", subject: Optional[str] = None, max_retries: int = 3) -> List[Experience]:
        """
        Extracts atomic experiences from (problem, solution).
        Returns a list of Experience objects.
        """
        system_prompt = Config.INITIALIZER_SYSTEM_PROMPT.replace("{max_tokens}", os.environ.get("EXP_MAX_TOKENS", "1024"))
        prompt = Config.INITIALIZER_USER_PROMPT.format(problem=problem, solution=solution)
        
        for attempt in range(max_retries):
            with LOG_LOCK:
                logging.info(f"[{item_id}] Initializer: Mining experiences (attempt {attempt+1}/{max_retries})...")
                logging.info(f"[{item_id}] --- INITIALIZER PROMPT ---")
                logging.info(f"[{item_id}] SYSTEM PROMPT: {system_prompt}")
                logging.info(f"[{item_id}] USER PROMPT: {prompt}")
                
            start_time = time.time()
            try:
                response = self.llm.invoke([
                    ("system", system_prompt),
                    ("user", prompt)
                ])  
                
                latency = time.time() - start_time
                
                with LOG_LOCK:
                    logging.info(f"[{item_id}] --- INITIALIZER RESPONSE ---")
                    logging.info(f"[{item_id}] RESPONSE CONTENT: {response.content}")
                    logging.info(f"[{item_id}] Initializer latency: {latency:.2f}s")

                json_str = self._extract_json(response.content)
                if not json_str:
                    raise ValueError("Empty JSON extracted")

                data = json.loads(json_str)
                
                if not isinstance(data, list):
                    if isinstance(data, dict):
                        data = [data]
                    else:
                        raise ValueError(f"Expected list or dict, got {type(data)}")

                experiences = []
                default_subject = subject if subject else "General"
                for item in data:
                    exp = Experience(
                        condition=item.get("condition", ""),
                        strategy=item.get("strategy", ""),
                        source_id=[item_id],
                        created_by_agent="Pi_init",
                        subject=item.get("subject") or default_subject
                    )
                    experiences.append(exp)
                
                with LOG_LOCK:
                    logging.info(f"[{item_id}] Initializer extracted {len(experiences)} experiences | Latency: {latency:.2f}s")
                return experiences
            except Exception as e:
                with LOG_LOCK:
                    logging.error(f"[{item_id}] Initializer failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return []

    def batch_run(self, items: List[Dict[str, Any]], max_workers: int = None) -> List[Experience]:
        """
        Batch extracts experiences from multiple items in parallel.
        """
        max_workers = max_workers or Config.MAX_WORKERS
        all_experiences = []
        
        logging.info(f"Starting batch initialization with {len(items)} items and {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.run, 
                    item["problem"], 
                    item["solution"], 
                    item_id=item.get("item_id", f"init_{i}"),
                    subject=item.get("subject") or item.get("subject")
                ): item 
                for i, item in enumerate(items)
            }
            
            for future in tqdm(as_completed(futures), total=len(items), desc="Initializing Pool"):
                item_exps = future.result()
                all_experiences.extend(item_exps)
        
        with LOG_LOCK:
            logging.info(f"Batch initialization complete. Extracted {len(all_experiences)} total experiences.")
        return all_experiences

def generate_embeddings(
    input_source: List[str],
    embedding_model: str,
    output_path: str,
    batch_size: int = 100,
    max_workers: int = 5,
    task_name: str = "Problem",
    ids: List[str] = None
) -> str:
    """
    Unified function to generate embeddings for a list of strings.
    
    Args:
        input_source: List of strings to embed (e.g. problem texts or experience conditions)
        embedding_model: Embedding model name
        output_path: Path to save the .npz file
        batch_size: Batch size for API
        max_workers: Parallel workers
        task_name: Name for logging (e.g., "Problem", "Experience")
        ids: Optional list of IDs corresponding to input_source.
        
    Returns:
        Path to the saved file
    """
    if not input_source:
        logging.warning(f"Empty input list for {task_name} embedding.")
        return ""

    # 1. Check if output already exists
    if os.path.exists(output_path):
        logging.info(f"Embeddings matrix already exists at {output_path}. Skipping.")
        return output_path

    # 2. Core embedding logic
    logging.info(f"Generating embeddings for {len(input_source)} {task_name} items using {embedding_model}...")
    emb_model = get_embeddings_model(embedding_model)
    
    def process_batch(batch_data):
        start_idx, batch = batch_data
        # Sanitize batch
        batch = [str(t) or "[EMPTY]" for t in batch]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return start_idx, emb_model.embed_documents(batch)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    raise e

    try:
        batches = [(i, input_source[i:i+batch_size]) for i in range(0, len(input_source), batch_size)]
        all_embeddings_dict = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_batch, b): b[0] for b in batches}
            with tqdm(total=len(input_source), desc=f"{task_name} Embedding Progress") as pbar:
                for future in as_completed(futures):
                    start_idx, batch_embeddings = future.result()
                    all_embeddings_dict[start_idx] = batch_embeddings
                    pbar.update(len(batch_embeddings))
        
        # Assemble in correct order
        sorted_keys = sorted(all_embeddings_dict.keys())
        all_embeddings = []
        for k in sorted_keys:
            all_embeddings.extend(all_embeddings_dict[k])
            
        matrix = np.array(all_embeddings)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if ids is None:
            logging.warning("Saving .npz without IDs provided. Only 'embeddings' key will be saved.")
            np.savez_compressed(output_path, embeddings=matrix)
        else:
            if len(ids) != len(matrix):
                 logging.warning(f"ID count ({len(ids)}) != Matrix rows ({len(matrix)}). Saving anyway.")
            np.savez_compressed(output_path, embeddings=matrix, ids=np.array(ids))
        logging.info(f"Saved {task_name} vector matrix ({matrix.shape}) and IDs to {output_path}")
            
        return output_path
        
    except Exception as e:
        logging.error(f"Failed to generate {task_name} embeddings: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="TSGD Initializer Agent")
    parser.add_argument("--initializer_model", type=str, default="grok-4-1-fast-non-reasoning", help="Model name for mining")
    parser.add_argument("--initializer_temperature", type=float, default=0.0, help="Temperature for initializer model")
    parser.add_argument("--input_path", type=str, required=True, help="Dataset name/path for mining")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers")
    parser.add_argument("--experience_path", type=str, help="Path to save mined experiences")
    parser.add_argument("--project_name", type=str, default="explearn_init", help="Phoenix project name")
    parser.add_argument("--log_file", type=str, help="Path to custom log file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-large", help="Embedding model to use")
    parser.add_argument("--embedding_path", type=str, help="Path to save/load problem embeddings")

    # Other potential overrides
    parser.add_argument("--temperature", type=float, help="LLM temperature")
    parser.add_argument("--max_tokens", type=int, help="LLM max tokens")
    
    args = parser.parse_args()

    # 1. Setup Environment
    Config.update(vars(args))
    if args.initializer_model:
        os.environ["INITIALIZER_MODEL"] = args.initializer_model
    if args.initializer_temperature is not None:
        os.environ["INITIALIZER_TEMPERATURE"] = str(args.initializer_temperature)
    if args.max_tokens is not None:
        os.environ["MAX_TOKENS"] = str(args.max_tokens)
    if args.max_workers is not None:
        os.environ["MAX_WORKERS"] = str(args.max_workers)
        os.environ["OTEL_PYTHON_OTLP_HTTP_MAX_POOL_SIZE"] = str(args.max_workers)
        os.environ["OTEL_EXPORTER_OTLP_HTTP_MAX_CONNECTIONS"] = str(args.max_workers)

    # 2. Setup Logging
    setup_logging(log_file=args.log_file)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 3. Setup Tracing
    setup_phoenix(project_name=args.project_name)

    # 4. Load Data
    logging.info(f"Loading dataset: {args.input_path}")
    data = DataLoader.load_dataset(args.input_path, split="train")
    
    if args.max_samples:
        data = DataLoader.partition_data(data, args.max_samples)

    # 5. Run Initialization
    logging.info(f"Running Initializer Agent on {len(data)} samples...")
    
    max_workers = args.max_workers or int(os.environ.get("MAX_WORKERS", Config.MAX_WORKERS))
    initializer = InitAgent(config=vars(args))
    raw_exps = initializer.batch_run(data, max_workers=max_workers)
    
    # 6. Save to ExperiencePool
    pool = ExperiencePool()
    
    # Manually add to registry to avoid computing embeddings for experiences
    for exp in raw_exps:
        pool.registry[exp.id] = exp
    
    # Determine output directory
    if args.experience_path:
        # If it's an existing directory, use it
        if os.path.isdir(args.experience_path):
            output_dir = args.experience_path
        # If it's a file path (has extension) and not an existing directory
        elif os.path.splitext(args.experience_path)[1] and not args.experience_path.endswith('/'):
            # Check if the extension is a common model version or part of a path (like .1-fast)
            # instead of a typical file extension like .jsonl, .json, .csv
            ext = os.path.splitext(args.experience_path)[1].lower()
            if ext in ['.jsonl', '.json', '.csv', '.txt', '.npz']:
                output_dir = os.path.dirname(args.experience_path)
            else:
                # If it's an unknown extension, it's likely part of a directory name (e.g. model version)
                output_dir = args.experience_path
        else:
            output_dir = args.experience_path
    else:
        output_dir = "data/exp_library"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # File 2: Experience Library
    exp_path = os.path.join(output_dir, "experience_pool.jsonl")
    pool.save(exp_path, save_index=False)
    
    # File 1 & 3: Problem Index and Metadata
    dataset_basename = os.path.splitext(os.path.basename(args.input_path))[0]
    
    meta_path = os.path.join(output_dir, f"question_meta.json")
    experience_idx = os.path.join(output_dir, f"experience_idx.npz")
    
    logging.info(f"Building problem index metadata for {dataset_basename}...")
    
    problem_ids = []
    problem_texts = []
    problem_meta = {}
    
    # Group experiences by source_id for fast lookup
    exp_by_source = {}
    for exp in pool.experiences:
        for sid in exp.source_id:
            if sid not in exp_by_source:
                exp_by_source[sid] = []
            exp_by_source[sid].append(exp.id)
            
    # Iterate over original data to ensure all problems are indexed
    for i, item in enumerate(data):
        # Use the same logic as batch_run to get item_id
        pid = item.get("item_id", f"init_{i}")
        problem_text = item.get("problem", "")
        
        problem_ids.append(pid)
        problem_texts.append(problem_text)
            
        problem_meta[pid] = {
            "problem": problem_text,
            "experience_ids": exp_by_source.get(pid, [])
        }
    
    # Save metadata
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(problem_meta, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved problem metadata to {meta_path}")

    # Generate problem embeddings
    problem_idx_path = os.path.join(os.path.dirname(args.input_path) or ".", f"{dataset_basename}_{args.embedding_model}_idx.npz")
    
    generate_embeddings(
        problem_texts, 
        args.embedding_model,
        problem_idx_path,
        task_name="Problem",
        ids=problem_ids
    )
    
    # Save experience index
    exp_conditions = [exp.condition or "[EMPTY]" for exp in pool.experiences]
    generate_embeddings(
        exp_conditions,
        args.embedding_model,
        experience_idx,
        max_workers=max_workers,
        task_name="Experience"
    )
    
    logging.info(f"Initialization complete. All files saved to {output_dir}")

if __name__ == "__main__":
    main()
