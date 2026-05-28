import logging
from typing import List, Dict, Any, Callable, Optional
from src.tools.utils import LOG_LOCK, Config
from src.tools.grader import extract_answer, math_equal
from src.agents.base_agent import BaseAgent

def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculates the pass@k metric.
    """
    if n - c < k:
        return 1.0
    
    import math
    def combinations(n, k):
        if k < 0 or k > n: return 0
        if k == 0 or k == n: return 1
        if k > n // 2: k = n - k
        
        numerator = 1
        for i in range(k):
            numerator = numerator * (n - i) // (i + 1)
        return numerator

    return 1.0 - combinations(n - c, k) / combinations(n, k)

class EvaluatorAgent(BaseAgent):
    """
    Pure Evaluator Agent: Focuses on assessing a prediction against ground truth.
    Input: (prediction, ground_truth)
    Output: (is_correct, reason)
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.debug = self.config.get("debug", False)

    def run(self, prediction: str, ground_truth: str, item_id: str = "N/A", problem: str = "", logger: Optional[Callable] = None) -> Any:
        return self.assess_prediction(prediction, ground_truth, item_id, problem, logger)

    def assess_prediction(
        self, 
        prediction: str, 
        ground_truth: str, 
        item_id: str = "N/A", 
        problem: str = "", 
        logger: Optional[Callable] = None
    ) -> tuple[bool, str]:
        """
        Assesses a prediction against ground truth using robust extraction and math comparison.
        """
        def log_func(msg):
            if logger: 
                logger(msg)
            else:
                with LOG_LOCK: 
                    logging.info(msg)

        # 1. Pre-check for errors
        if any(err in prediction for err in ["Error: Connection error", "Error: API error"]):
            return False, "connection_error"

        # 2. Answer Extraction
        extracted_prediction = extract_answer(prediction)
        gt_answer = extract_answer(ground_truth) or str(ground_truth).strip()
        
        if self.debug:
            log_func(f"[{item_id}] [Debug] Problem ID: {item_id}")
            if problem:
                log_func(f"[{item_id}] [Debug] Original Problem: {problem}")
            log_func(f"[{item_id}] [Debug] Original Prediction: {prediction}")
            log_func(f"[{item_id}] [Debug] Extracted Prediction: {extracted_prediction}")
            log_func(f"[{item_id}] [Debug] Ground Truth Answer: {gt_answer}")
        
        # 3. Rule-based Comparison
        try:
            if math_equal(extracted_prediction, gt_answer):
                if self.debug:
                    log_func(f"[{item_id}] [Debug] Result: CORRECT (Method: math_equal)")
                return True, "correct"
        except Exception as e:
            if self.debug: 
                log_func(f"[{item_id}] [Debug] math_equal error: {e}")
        
        if not extracted_prediction:
            if self.debug:
                log_func(f"[{item_id}] [Debug] Result: WRONG (extraction_failed)")
            return False, "extraction_failed"
            
        if self.debug:
            log_func(f"[{item_id}] [Debug] Result: WRONG (mismatch)")
        return False, "mismatch"
