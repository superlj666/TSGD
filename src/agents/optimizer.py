import json
import logging
import os
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.agents.base_agent import BaseAgent
from src.core.experience_pool import Experience
from src.tools.utils import get_chat_model, Config, LOG_LOCK

class OptimizerAgent(BaseAgent):
    """
    Optimizer Agent (Pi_grad):
    Analyzes errors and generates textual gradients (Add/Edit/Delete actions).
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.llm = self._init_llm("optimizer")
        
    def run_single(self, error_case: Dict[str, Any], case_id: int = 0) -> List[Dict[str, Any]]:
        """
        Analyzes a single error case and returns suggested actions.
        """
        exp_str = ""
        used_exps = error_case.get("used_experiences", [])
        if used_exps:
            exp_str = "### Used Experiences:\n"
            for e in used_exps:
                warning_str = f"\nWarning: {e.warning}" if getattr(e, "warning", "") else ""
                exp_str += f"ID: {e.id}\nCondition: {e.condition}\nStrategy: {e.strategy}{warning_str}\n\n"
        
        system_prompt = Config.OPTIMIZER_SYSTEM_PROMPT.replace("{max_tokens}", os.environ.get("EXP_MAX_TOKENS", "1024"))
        prompt = Config.OPTIMIZER_USER_PROMPT.format(
            problem=error_case.get('problem'),
            gt_solution=error_case.get('gt_solution'),
            error_cases=error_case.get('error_cases'),
            used_experiences=exp_str
        )
        
        item_id = error_case.get("item_id", "N/A")
        with LOG_LOCK:
            logging.info(f"[Case {case_id}] [Item {item_id}] --- OPTIMIZER REQUEST ---")
            logging.debug(f"[Case {case_id}] [Item {item_id}] SYSTEM PROMPT:\n{system_prompt}")
            logging.debug(f"[Case {case_id}] [Item {item_id}] USER PROMPT:\n{prompt}")
        
        start_time = time.time()
        try:
            response = self.llm.invoke([
                ("system", system_prompt),
                ("user", prompt)
            ])
            
            latency = time.time() - start_time
            content = response.content
            
            with LOG_LOCK:
                logging.info(f"[Case {case_id}] [Item {item_id}] --- OPTIMIZER RESPONSE (latency: {latency:.2f}s) ---\n{content}")
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
                
            actions = json.loads(content)
            if isinstance(actions, dict):
                actions = [actions]
            
            return actions
        except Exception as e:
            with LOG_LOCK:
                logging.error(f"[Case {case_id}] [Item {item_id}] Optimizer failed: {e}")
            return []

    def run(self, errors: List[Dict[str, Any]], max_workers: int = None) -> List[Dict[str, Any]]:
        """
        Analyzes a batch of errors and returns a list of suggested actions.
        Input: errors: List of dicts, each containing:
               - problem
               - gt_solution
               - pred_trajectory
               - used_experiences (List[Experience])
        """
        if not errors:
            return []

        max_workers = max_workers or self.config.get("max_workers", Config.MAX_WORKERS)
        all_actions = []
        
        # Generate prefix for logging
        item_ids = sorted(list(set(str(e.get("item_id", "N/A")) for e in errors)))
        if not item_ids:
            prefix = ""
        elif len(item_ids) == 1:
            prefix = f"[{item_ids[0]}]"
        else:
            prefix = f"[{','.join(item_ids[:3])}{'...' if len(item_ids)>3 else ''}]"
        
        logging.info(f"{prefix} Starting parallel optimization with {len(errors)} errors and {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run_single, err, i+1): i 
                for i, err in enumerate(errors)
            }
            
            for future in tqdm(as_completed(futures), total=len(errors), desc="Optimizing"):
                actions = future.result()
                all_actions.extend(actions)
        
        logging.info(f"{prefix} Optimization complete. Suggested {len(all_actions)} total actions.")
        return all_actions
