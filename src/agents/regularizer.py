import json
import logging
import os
import argparse
import sys
import re

# Add the project root to sys.path to allow importing from src.tools.utils etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from typing import List, Dict, Any
from src.agents.base_agent import BaseAgent
from src.core.experience_pool import ExperiencePool, Experience
from src.tools.utils import get_chat_model, Config, load_prompts, setup_logging, setup_phoenix

class RegularizerAgent(BaseAgent):
    """
    Regularizer Agent (Pi_reg):
    Maintains the experience library budget by merging and pruning.
    """
    def __init__(self, experience_pool: ExperiencePool, config: Dict[str, Any] = None):
        super().__init__(config)
        self.experience_pool = experience_pool
        self.llm = self._init_llm("regularizer")
        self.prompts = load_prompts()
        self.problem_embeddings = None
        self.experience_embeddings = None
        self.experience_ids = []
        
        # Load problem embeddings if path provided
        if config and config.get("problem_embedding_path"):
            if config.get("meta_path") or config.get("problem_ids"):
                self._load_problem_embeddings(
                    config["problem_embedding_path"], 
                    config.get("meta_path"),
                    config.get("problem_ids")
                )
        
        # Load experience embeddings if path provided
        if config and config.get("experience_embedding_path"):
            self._load_experience_embeddings(config["experience_embedding_path"])

    def _init_llm(self, agent_name: str):
        """
        Overriding _init_llm to ensure higher max_tokens for regularization tasks.
        """
        llm = super()._init_llm(agent_name)
        # Ensure max_tokens is at least 8192 for regularizer to avoid truncation
        if hasattr(llm, "max_tokens") and (llm.max_tokens is None or llm.max_tokens < 8192):
            llm.max_tokens = 8192
            logging.info(f"[{agent_name.capitalize()}] Boosted max_tokens to {llm.max_tokens}")
        return llm

    def _load_problem_embeddings(self, embedding_path: str, meta_path: str = None, problem_ids: List[str] = None):
        """
        Loads problem embeddings from .npz and maps them to problem IDs using meta.json or provided list.
        """
        import numpy as np
        try:
            # Enforce .npz
            if os.path.exists(embedding_path):
                data = np.load(embedding_path)
                embeddings = data["embeddings"]
                if "ids" in data and problem_ids is None:
                    problem_ids = data["ids"].tolist()
                    logging.info(f"Loaded {len(problem_ids)} problem IDs from .npz")
            else:
                logging.error(f"Problem embedding file not found: {embedding_path}")
                self.problem_embeddings = None
                return
            
            if problem_ids is None:
                if meta_path:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    # Assume order in meta_data keys matches embeddings rows
                    problem_ids = list(meta_data.keys())
                else:
                    logging.warning("No meta_path or problem_ids provided for problem embeddings.")
                    return

            if len(problem_ids) != len(embeddings):
                 logging.warning(f"Embedding count {len(embeddings)} != Problem ID count {len(problem_ids)}. Problem embedding clustering might be misaligned.")
            
            self.problem_embeddings = {}
            for i, pid in enumerate(problem_ids):
                if i < len(embeddings):
                    self.problem_embeddings[pid] = embeddings[i]
            
            logging.info(f"Loaded {len(self.problem_embeddings)} problem embeddings.")
            
        except Exception as e:
            logging.error(f"Failed to load problem embeddings: {e}")
            self.problem_embeddings = None

    def _load_experience_embeddings(self, embedding_path: str):
        """
        Loads experience condition embeddings from .npz.
        """
        import numpy as np
        try:
            # Enforce .npz
            if not os.path.exists(embedding_path):
                 logging.warning(f"Experience embedding file not found: {embedding_path}")
                 self.experience_embeddings = None
                 return

            data = np.load(embedding_path)
            # Support both raw array (old style if any) or structured .npz
            if "embeddings" in data:
                self.experience_embeddings = data["embeddings"]
            else:
                # Fallback if it was somehow saved as just array in .npz (unlikely but safe)
                # This assumes data is array-like if keys not present, but np.load returns NpzFile
                # Actually np.savez saves with keys 'arr_0' if not specified.
                # But our initializer uses named args: np.savez(..., embeddings=..., ids=...)
                # So 'embeddings' key should be there.
                if "arr_0" in data:
                     self.experience_embeddings = data["arr_0"]
                else:
                     logging.error(f"Could not find 'embeddings' or 'arr_0' in {embedding_path}")
                     self.experience_embeddings = None
                     return

            # Map them by index to experience IDs in the pool at the time of loading
            self.experience_ids = [exp.id for exp in self.experience_pool.experiences]
            
            # If .npz has IDs, we can verify alignment
            if "ids" in data:
                loaded_ids = data["ids"].tolist()
                if len(loaded_ids) == len(self.experience_ids):
                     # Optional: strict check
                     # if loaded_ids != self.experience_ids:
                     #    logging.warning("Loaded experience IDs do not match current pool IDs!")
                     pass
                else:
                     logging.warning(f"Loaded ID count {len(loaded_ids)} != Pool ID count {len(self.experience_ids)}")

            if len(self.experience_ids) != len(self.experience_embeddings):
                 logging.warning(f"Experience count {len(self.experience_ids)} != Embedding count {len(self.experience_embeddings)}. Experience embedding clustering might be misaligned.")
            
            logging.info(f"Loaded experience matrix: {self.experience_embeddings.shape}")
        except Exception as e:
            logging.error(f"Failed to load experience embeddings: {e}")
            self.experience_embeddings = None

    def _robust_json_parse(self, text: str) -> List[Dict[str, Any]]:
        """
        Robustly extracts and parses JSON list from LLM output.
        Handles truncation and unescaped characters.
        """
        if not text or not text.strip():
            return []

        content = text.strip()

        # 1. Handle Markdown Code Blocks First
        # Regex to capture content inside ```json ... ``` or just ``` ... ```
        # Flags: DOTALL to match newlines
        import re
        code_block_pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
        match = code_block_pattern.search(content)
        if match:
            content = match.group(1).strip()
        else:
            # If no code block found, checking for single trailing ``` 
            # (common if header was sliced or malformed)
            if "```" in content:
                content = content.replace("```", "").strip()

        # 2. Locate JSON list boundaries
        start_idx = content.find("[")
        if start_idx != -1:
            content = content[start_idx:]
        
        last_bracket = content.rfind("]")
        if last_bracket != -1:
            content = content[:last_bracket+1]

        # 3. Attempt parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
             # If simple load fails, try recovery strategies
             pass

        # 4. Handle Truncation: Attempt to repair truncated JSON list
        logging.warning("JSON parsing failed, attempting recovery for truncated output...")
        
        # Try to find the last complete object
        # The most reliable way is to find the last "}" followed by optional whitespace and ","
        # or just the last "}"
        
        # Try finding the last "}"
        last_brace = content.rfind("}")
        if last_brace != -1:
            repaired = content[:last_brace + 1]
            # Ensure it starts with [
            if not repaired.startswith("["):
                repaired = "[" + repaired
            # Ensure it ends with ]
            if not repaired.endswith("]"):
                repaired = repaired + "]"
            
            try:
                data = json.loads(repaired)
                logging.info(f"Successfully recovered {len(data)} items from truncated JSON.")
                return data
            except Exception as e:
                logging.debug(f"Recovery method 1 failed: {e}")

        # 5. Regex Fallback: Extract all valid-looking objects
        # This is more aggressive and handles cases where even the last object is partially broken
        try:
            # Find all strings that look like JSON objects: {...}
            # We use a non-greedy match that tries to balance braces (roughly)
            # A simpler regex that just finds things between { and }
            import re
            objects = re.findall(r'\{[^{}]*\}', content)
            recovered = []
            for obj_str in objects:
                try:
                    obj = json.loads(obj_str)
                    if isinstance(obj, dict):
                        recovered.append(obj)
                except:
                    continue
            
            if recovered:
                logging.info(f"Recovered {len(recovered)} items using regex fallback.")
                return recovered
        except Exception as e:
            logging.error(f"Regex recovery failed: {e}")
                
        return []

    def run(self, total_limit: int = 300, subject_limit: int = 100, max_tokens: int = 1024, mode: str = "problem") -> List[Dict[str, Any]]:
        """
        Maintains the experience library budget by merging and pruning.
        Budget includes:
        - total_limit: Maximum total number of experiences.
        - subject_limit: Maximum experiences per subject.
        - max_tokens: Maximum character length per experience (condition + strategy).
        - mode: 'problem' or 'condition' for clustering.
        """
        current_size = len(self.experience_pool.experiences)
        logging.info(f"Regularizer: Starting regularization (Mode={mode}). Current size: {current_size}, Budget: total={total_limit}, subject={subject_limit}, tokens={max_tokens}")

        actions = []

        # 1. Per-subject regularization using LLM
        self._regularize_by_subjects(subject_limit, max_tokens, mode=mode)

        # 2. Global utility-based pruning
        if len(self.experience_pool.experiences) > total_limit:
            pruned_ids = self._global_utility_pruning(total_limit)
            for pid in pruned_ids:
                actions.append({"action": "DELETE", "id": pid, "warning": "Global utility pruning"})

        # self.experience_pool._update_matrix()  # Disable matrix update for now
        logging.info(f"Regularizer: Finished. Final size: {len(self.experience_pool.experiences)}")
        return actions

    def _regularize_by_subjects(self, subject_limit: int, max_tokens: int, mode: str = "problem"):
        """
        Groups experiences by subject and uses clustering + staged LLM to merge/refine each group.
        """
        from collections import defaultdict
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        subject_groups = defaultdict(list)
        for exp in self.experience_pool.experiences:
            subject_groups[exp.subject].append(exp)
        
        prompt_config = self.prompts.get("regularizer", {})
        system_prompt_template = prompt_config.get("system_prompt", "")
        user_prompt_template = prompt_config.get("user_prompt", "")
        
        if not system_prompt_template or not user_prompt_template:
            logging.warning("Regularizer: Regularizer prompts not found. Skipping LLM-based subject regularization.")
            return

        for subject, group in subject_groups.items():
            if len(group) <= 1:
                continue
            
            subject_name = subject if subject else "General"
            logging.info(f"Regularizer: Processing subject '{subject_name}' with {len(group)} items.")
            
            # 1. Clustering based on SUBJECT_LIMIT
            num_clusters = min(10, int(len(group) / subject_limit))
            # Fallback check: if mode is problem but embeddings missing, log warning
            if mode == "problem" and not self.problem_embeddings:
                logging.warning(f"  > Mode is 'problem' but 'problem_embeddings' is empty. Falling back to experience-based clustering.")
                
            if mode == "problem" and self.problem_embeddings:
                 clusters = self._k_means_clustering_by_problem(group, num_clusters)
                 logging.info(f"  > Keyword '{subject_name}' clustered into {len(clusters)} clusters using PROBLEM embeddings.")
            else:
                 clusters = self._k_means_clustering(group, num_clusters)
                 logging.info(f"  > Keyword '{subject_name}' clustered into {len(clusters)} clusters using EXPERIENCE embeddings.")
            
            # 2. Sort clusters by size (descending) and process ALL clusters
            clusters.sort(key=len, reverse=True)
            
            refined_experiences_data = []
            
            # Process each cluster
            for i, cluster in enumerate(clusters):
                if len(cluster) < max(10, subject_limit*0.8):
                    logging.info(f"  > Cluster {i+1}/{len(clusters)} size ({len(cluster)}) < {max(10, subject_limit*0.8)}. Skipping LLM regularizer and keeping all.")
                    cluster_refined = [{
                        "condition": e.condition,
                        "strategy": e.strategy,
                        "warning": getattr(e, "warning", ""),
                        "source_id": e.source_id,
                        "subject": e.subject or subject_name,
                        "created_by_agent": e.created_by_agent
                    } for e in cluster]
                else:
                    logging.info(f"  > Processing Cluster {i+1}/{len(clusters)} (Size: {len(cluster)})")
                    cluster_refined = self._call_llm_regularizer(
                        subject_name, 
                        cluster, 
                        subject_limit, # Per-cluster target limit
                        max_tokens,
                        system_prompt_template,
                        user_prompt_template
                    )
                refined_experiences_data.extend(cluster_refined)
                logging.debug(f"  > Cluster {i+1} refined data: {json.dumps(cluster_refined, indent=2, ensure_ascii=False)}")
            
            # 3. Final refinement if total from clusters exceeds subject_limit
            final_subject_experiences = []
            if len(refined_experiences_data) > subject_limit:
                logging.info(f"  > Final merging for subject '{subject_name}': {len(refined_experiences_data)} -> {subject_limit}")
                # Convert data back to temporary Experience objects for _call_llm_regularizer
                temp_exps = [
                    Experience(
                        condition=item.get("condition", ""),
                        strategy=item.get("strategy", ""),
                        warning=item.get("warning") or "",
                        subject=subject_name,
                        source_id=item.get("source_id", []),
                        created_by_agent="Pi_reg_temp"
                    ) for item in refined_experiences_data
                ]
                final_data = self._call_llm_regularizer(
                    subject_name,
                    temp_exps,
                    subject_limit,
                    max_tokens,
                    system_prompt_template,
                    user_prompt_template
                )
                final_subject_experiences = final_data[:subject_limit]
            else:
                final_subject_experiences = refined_experiences_data

            # Update the pool: Delete old and add new
            for e in group:
                self.experience_pool.delete(e.id)
            
            for item in final_subject_experiences:
                new_exp = Experience(
                    condition=item.get("condition", ""),
                    strategy=item.get("strategy", ""),
                    warning=item.get("warning") or "",
                    subject=item.get("subject") or subject_name,
                    source_id=item.get("source_id", []),
                    created_by_agent=item.get("created_by_agent", "Pi_reg")
                )
                logging.debug(f"  > Adding refined experience: {json.dumps(new_exp.to_dict(), indent=2, ensure_ascii=False)}")
                self.experience_pool.add(new_exp)
                
                # Link to source problems for retrieval - REMOVED as per request
                # for sid in new_exp.source_id:
                #    self.experience_pool.link_experience_to_problem(new_exp.id, sid)
            
            logging.info(f"  > Keyword '{subject_name}' regularization complete: {len(group)} -> {len(final_subject_experiences)} items.")

    def _k_means_clustering_by_problem(self, experiences: List[Experience], n_clusters: int) -> List[List[Experience]]:
        import numpy as np
        from sklearn.cluster import KMeans
        
        if not experiences:
            return []
        if n_clusters <= 1:
            return [experiences]
            
        # Collect embeddings for each experience
        embeddings = []
        valid_indices = []
        
        for i, exp in enumerate(experiences):
            # Get embedding from source_id
            # Assuming first source_id is the primary one or they are similar
            emb = None
            for sid in exp.source_id:
                if sid in self.problem_embeddings:
                    emb = self.problem_embeddings[sid]
                    break
            
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(i)
            else:
                pass

        if not embeddings:
             logging.warning("No problem embeddings found for any experience in group. Falling back to simple split.")
             return [experiences]

        embeddings = np.array(embeddings)
        
        # If we have fewer embeddings than n_clusters, reduce n_clusters
        if len(embeddings) < n_clusters:
            n_clusters = len(embeddings)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        
        clusters = [[] for _ in range(n_clusters)]
        
        # Assign clustered items
        for idx, label in enumerate(labels):
            exp_idx = valid_indices[idx]
            clusters[label].append(experiences[exp_idx])
            
        # Assign unclustered items (those without problem embeddings) to the first cluster
        unclustered_indices = set(range(len(experiences))) - set(valid_indices)
        if unclustered_indices:
            logging.warning(f"{len(unclustered_indices)} experiences had no problem embedding. Assigning to cluster 0.")
            for idx in unclustered_indices:
                if clusters:
                    clusters[0].append(experiences[idx])
                else:
                    clusters.append([experiences[idx]])
                    
        return [c for c in clusters if c]

    def _k_means_clustering(self, experiences: List[Experience], n_clusters: int) -> List[List[Experience]]:
        """
        Clusters experiences using K-Means.
        """
        import numpy as np
        from sklearn.cluster import KMeans
        
        if not experiences:
            return []
        if n_clusters <= 1:
            return [experiences]
            
        # Get embeddings
        embeddings = []
        valid_indices = []
        
        # 1. Try to use pre-loaded embeddings first
        if self.experience_embeddings is not None:
            # Create a lookup map for faster access
            id_to_idx = {eid: i for i, eid in enumerate(self.experience_ids)}
            for i, exp in enumerate(experiences):
                if exp.id in id_to_idx:
                    embeddings.append(self.experience_embeddings[id_to_idx[exp.id]])
                    valid_indices.append(i)
        
        # 2. For items without pre-loaded embeddings, calculate on the fly (or if none loaded)
        if len(embeddings) < len(experiences):
            remaining_indices = set(range(len(experiences))) - set(valid_indices)
            logging.info(f"  > Calculating embeddings on-the-fly for {len(remaining_indices)} items.")
            for i in sorted(list(remaining_indices)):
                exp = experiences[i]
                emb = self.experience_pool._get_condition_embedding(exp.condition)
                embeddings.append(emb)
                valid_indices.append(i)
        
        embeddings = np.array(embeddings)
        
        # Fit K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        
        clusters = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(labels):
            exp_idx = valid_indices[idx]
            clusters[label].append(experiences[exp_idx])
            
        # Filter out empty clusters if any
        return [c for c in clusters if c]

    def _call_llm_regularizer(self, subject, group, limit, max_tokens, sys_tmpl, user_tmpl):
        """
        Helper to call LLM for a specific group of experiences.
        """
        exp_list_for_llm = []
        for e in group:
            exp_list_for_llm.append({
                "id": e.id,
                "condition": e.condition,
                "strategy": e.strategy,
                "source_id": e.source_id,
                "warning": e.warning
            })
        
        system_prompt = sys_tmpl.replace("{max_tokens}", str(max_tokens))
        system_prompt = system_prompt.replace("{subject_limit}", str(limit))
        user_prompt = user_tmpl.replace("{experience_list}", json.dumps(exp_list_for_llm, ensure_ascii=False))
        user_prompt = user_prompt.replace("{subject}", str(subject))
        user_prompt = user_prompt.replace("{subject_limit}", str(limit))
        
        logging.info(f"--- LLM Regularizer PROMPT (Keyword: {subject}, Items: {len(group)}) ---")
        logging.info(f"SYSTEM PROMPT:\n{system_prompt}")
        logging.info(f"USER PROMPT:\n{user_prompt}")
        logging.info("-" * 50)

        try:
            response = self.llm.invoke([
                ("system", system_prompt),
                ("user", user_prompt)
            ])
            
            logging.info(f"--- LLM Regularizer RESPONSE (Keyword: {subject}) ---")
            logging.info(f"CONTENT:\n{response.content}")
            logging.info("-" * 50)

            data = self._robust_json_parse(response.content)
            
            if not isinstance(data, list):
                return []
            
            # Validation and Fallback for source_id and subject
            for item in data:
                # 1. Ensure source_id is a list and not empty
                sid = item.get("source_id")
                if not sid or sid == "N/A" or sid == ["N/A"]:
                    # Try to find which input experience this refined one might have come from
                    # or just aggregate all source_ids from the group as a fallback
                    all_ids = []
                    for e in group:
                        if isinstance(e.source_id, list):
                            all_ids.extend(e.source_id)
                        else:
                            all_ids.append(str(e.source_id))
                    
                    # Ensure we have at least "N/A" if everything else failed
                    item["source_id"] = sorted(list(set([i for i in all_ids if i])))
                    if not item["source_id"]:
                        item["source_id"] = ["N/A"]
                        
                    logging.warning(f"  > Regularizer fallback: item missing source_id, assigned {item['source_id']} from group.")
                elif not isinstance(sid, list):
                    item["source_id"] = [str(sid)]
                else:
                    # Filter out any "N/A" or empty strings from the LLM-provided list
                    original_sid = list(sid)
                    item["source_id"] = [str(i) for i in sid if i and i != "N/A"]
                    if not item["source_id"]:
                        # If filtering left us empty, use the group fallback
                        all_ids = []
                        for e in group:
                            if isinstance(e.source_id, list):
                                all_ids.extend(e.source_id)
                            else:
                                all_ids.append(str(e.source_id))
                        item["source_id"] = sorted(list(set([i for i in all_ids if i])))
                        if not item["source_id"]:
                            item["source_id"] = ["N/A"]
                        logging.warning(f"  > Regularizer fallback: item had empty/invalid source_id list {original_sid}, assigned {item['source_id']} from group.")
                
                # 2. Ensure subject is present
                if "subject" not in item or not item["subject"] or item["subject"] == "N/A":
                    item["subject"] = subject
                    
            return data
        except Exception as e:
            logging.error(f"Error calling LLM regularizer for subject '{subject}': {e}")
            return [{"condition": e.condition, "strategy": e.strategy, "warning": getattr(e, "warning", ""), "source_id": e.source_id, "subject": getattr(e, "subject", subject)} for e in group[:limit]]

    def _global_utility_pruning(self, total_limit: int) -> List[str]:
        """
        Removes experiences with the lowest utility score until size <= total_limit.
        Returns list of pruned experience IDs.
        """
        pruned_ids = []
        if len(self.experience_pool.experiences) <= total_limit:
            return pruned_ids

        # Calculate utility for all
        utilities = []
        for exp in self.experience_pool.experiences:
            # Simple utility: (success_count + 1) / (failure_count + 1)
            # You can make this more complex (e.g., recency weighted)
            u = (exp.success_count + 1) / (exp.failure_count + 1)
            utilities.append((exp.id, u))
        
        # Sort by utility (ascending) - remove lowest first
        utilities.sort(key=lambda x: x[1])
        
        num_to_remove = len(self.experience_pool.experiences) - total_limit
        to_remove = utilities[:num_to_remove]
        
        for exp_id, u in to_remove:
            logging.info(f"Regularizer: Pruning low utility experience {exp_id} (Utility: {u:.2f})")
            self.experience_pool.delete(exp_id)
            pruned_ids.append(exp_id)
            
        return pruned_ids

    def _semantic_merge(self):
        # Kept for backward compatibility if needed, but not used in the new run() flow
        pass

def main():
    parser = argparse.ArgumentParser(description="TSGD Regularizer Agent")
    parser.add_argument("--regularizer_model", type=str, default="gpt-4o", help="Model name for regularization")
    parser.add_argument("--regularizer_temperature", type=float, default=0.0, help="Temperature for regularizer model")
    parser.add_argument("--experience_path", type=str, required=True, help="Path to input experience pool")
    parser.add_argument("--meta_path", type=str, help="Path to meta.json file for synchronization")
    parser.add_argument("--meta_output_path", type=str, help="Path to save updated meta.json file")
    parser.add_argument("--output_path", type=str, help="Path to save regularized pool")
    parser.add_argument("--total_limit", type=int, default=50, help="Total experience limit")
    parser.add_argument("--subject_limit", type=int, default=10, help="Per-subject experience limit")
    parser.add_argument("--mode", type=str, default="problem", choices=["problem", "condition"], help="Clustering mode: 'problem' (source problem embeddings) or 'condition' (experience condition embeddings)")
    parser.add_argument("--exp_max_tokens", type=int, default=1000, help="Max tokens per experience")
    parser.add_argument("--problem_embedding_path", type=str, help="Path to problem embeddings (.npz)")
    parser.add_argument("--experience_embedding_path", type=str, help="Path to experience condition embeddings (.npz)")
    parser.add_argument("--project_name", type=str, default="explearn_reg", help="Phoenix project name")
    parser.add_argument("--log_file", type=str, help="Path to custom log file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-large", help="Embedding model to use")
    
    # LLM Hyperparameters
    parser.add_argument("--temperature", type=float, help="LLM temperature")
    parser.add_argument("--max_tokens", type=int, help="LLM max tokens")
    
    args = parser.parse_args()

    # 1. Setup Environment
    if args.regularizer_model:
        os.environ["MODEL_NAME"] = args.regularizer_model
    if args.regularizer_temperature:
        os.environ["TEMPERATURE"] = str(args.regularizer_temperature)
    if args.max_tokens:
        os.environ["MAX_TOKENS"] = str(args.max_tokens)

    # 2. Setup Logging
    setup_logging(log_file=args.log_file)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 3. Setup Tracing
    setup_phoenix(project_name=args.project_name)

    # 4. Load Pool
    if not os.path.exists(args.experience_path):
        logging.error(f"Experience path {args.experience_path} not found.")
        return
        
    pool = ExperiencePool()
    pool.load(args.experience_path)
    logging.info(f"Loaded {len(pool.experiences)} experiences from {args.experience_path}")

    # 5. Run Regularization
    regularizer = RegularizerAgent(experience_pool=pool, config=vars(args))
    regularizer.run(
        total_limit=args.total_limit,
        subject_limit=args.subject_limit,
        max_tokens=args.exp_max_tokens,
        mode=args.mode
    )

    # 6. Save Pool
    output_path = args.output_path or args.experience_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pool.save(output_path)
    logging.info(f"Regularization complete. Saved pool to {output_path}")

    # 7. Update Meta File (if provided)
    if args.meta_path and os.path.exists(args.meta_path):
        logging.info(f"Updating meta file at {args.meta_path}...")
        try:
            with open(args.meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            # Rebuild PID -> ExpIDs mapping from the pool
            pid_to_exps = {}
            for exp in pool.experiences:
                for src_id in exp.source_id:
                    if src_id and src_id != "N/A":
                        if src_id not in pid_to_exps:
                            pid_to_exps[src_id] = []
                        pid_to_exps[src_id].append(exp.id)
            
            # Update meta_data
            updated_count = 0
            for pid, data in meta_data.items():
                new_exps = pid_to_exps.get(pid, [])
                # Only update if changed
                if set(data.get("experience_ids", [])) != set(new_exps):
                    data["experience_ids"] = new_exps
                    updated_count += 1
            
            # Save meta file
            save_meta_path = args.meta_output_path or args.meta_path
            with open(save_meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Updated {updated_count} entries in meta file. Saved to {save_meta_path}")
            
        except Exception as e:
            logging.error(f"Failed to update meta file: {e}")

if __name__ == "__main__":
    main()
