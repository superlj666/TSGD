import logging
import time
from typing import List, Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from src.tools.utils import Config, get_chat_model, LOG_LOCK, retry_with_exponential_backoff

from src.agents.base_agent import BaseAgent

class SolverAgent(BaseAgent):
    """
    Solver Agent (Pi_solver) that generates mathematical solutions.
    It uses the retrieved experiences to guide the reasoning process.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Use provided config or fallback to global Config
        self.llm = self._init_llm("solver")
        self.debug = self.config.get("debug", False)
        self.max_retries = self.config.get("max_retries", Config.MAX_RETRIES)

    def run(self, problem: str, context: List[Any] = [], item_id: str = "N/A") -> Dict[str, Any]:
        """
        Implementation of the abstract run method from BaseAgent.
        """
        return self.solve(problem, context, item_id)

    def solve(self, problem: str, context: List[Any] = [], item_id: str = "N/A") -> Dict[str, Any]:
        if context:
            # Use Experience-based Prompt
            context_str = ""
            for i, e in enumerate(context):
                warning_str = f"Warning: {e.warning}\n" if getattr(e, "warning", "") else ""
                context_str += f"Experience [{e.id}]:\n"
                context_str += f"Condition: {e.condition}\n"
                context_str += f"Strategy: {e.strategy}\n"
                context_str += f"{warning_str}\n"
            system_prompt_tmpl = Config.SOLVER_EXP_SYSTEM_PROMPT
            user_prompt_tmpl = Config.SOLVER_EXP_USER_PROMPT
            
        else:
            # Use Standard Prompt
            context_str = ""
            system_prompt_tmpl = Config.SOLVER_SYSTEM_PROMPT
            user_prompt_tmpl = Config.SOLVER_USER_PROMPT
            
        # Create a temporary prompt template for this call
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt_tmpl),
            ("user", user_prompt_tmpl)
        ])
        
        # Format inputs
        input_dict = {
            "problem": problem, 
            "library": context_str, 
        }
            
        start_time = time.time()
        
        # Prepare full prompt for logging
        try:
            full_system_prompt = system_prompt_tmpl.format(**input_dict) if "{library}" in system_prompt_tmpl else system_prompt_tmpl
        except:
            full_system_prompt = system_prompt_tmpl
            
        try:
            full_user_prompt = user_prompt_tmpl.format(**input_dict)
        except:
             full_user_prompt = user_prompt_tmpl.format(context="", problem=problem)

        with LOG_LOCK:
            logging.info(f"[{item_id}] --- SOLVER REQUEST ---")
            logging.info(f"[{item_id}] SYSTEM PROMPT:\n{full_system_prompt}")
            logging.info(f"[{item_id}] USER PROMPT:\n{full_user_prompt}")
        
        @retry_with_exponential_backoff(max_retries=self.max_retries)
        def _invoke_llm(input_dict, prompt_template, item_id="N/A"):
            chain = prompt_template | self.llm
            return chain.invoke(input_dict)

        try:
            response = _invoke_llm(input_dict, prompt_template, item_id=item_id)
            latency = time.time() - start_time
            
            token_usage = response.response_metadata.get("token_usage", {})
            content = response.content

            with LOG_LOCK:
                logging.info(f"[{item_id}] --- SOLVER RESPONSE ---")
                logging.info(f"[{item_id}] CONTENT:\n{content}")
                logging.info(f"[{item_id}] LATENCY: {latency:.2f}s | TOKENS: {token_usage.get('total_tokens', 0)}")

            # Extract used experience IDs if present
            used_exp_ids = []
            if "Used EXP:" in content:
                try:
                    import re
                    match = re.search(r"Used EXP: \{(.*?)\}", content)
                    if match:
                        ids_str = match.group(1)
                        used_exp_ids = [id_strip.strip() for id_strip in ids_str.split(",") if id_strip.strip()]
                except:
                    pass

            return {
                "prediction": content,
                "used_exp_ids": used_exp_ids,
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
                "latency": latency
            }
        except Exception as e:
            with LOG_LOCK:
                logging.error(f"[{item_id}] SOLVER ERROR: {str(e)}")
            return {
                "prediction": f"Error: {str(e)}",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency": time.time() - start_time
            }
