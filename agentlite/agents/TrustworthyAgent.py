from typing import List, Dict, Any
import csv
from utils import f1_score
from datetime import datetime
from dotenv import load_dotenv
from .BaseAgent import BaseAgent
from agentlite.commons import TaskPackage, AgentAct
from agentlite.actions import FinishAct
from agentlite.commons.AgentAct import ActObsChainType

# ----- Minimal TLM Setup -----------------
from cleanlab_tlm import TLM
# -----------------------------------------

load_dotenv()

class TrustworthyAgent(BaseAgent):
    """A TrustworthyAgent that tracks and scores LLM interactions for trustworthiness.
    
    This agent extends BaseAgent to add trustworthiness scoring functionality.
    It tracks all LLM interactions (prompts and responses) and saves them to CSV.
    
    Additional parameters:
        trust_score_file: str, optional
            Path to save trustworthiness scores.
        score_last_only: bool, optional
            Whether to only score the last Finish act.
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        llm: Any,
        actions: List[Any] = [],
        trust_score_file: str = None,
        score_last_only: bool = False,
        **kwargs
    ):
        # ---------------- Logger Setup (Optional) ----------------
        from agentlite.logging.terminal_logger import TrustworthyAgentLogger
        llm_model_name = getattr(llm, "llm_name", "unk")
        agent_arch = kwargs.get("agent_arch", "unk")
        logger = kwargs.pop('logger', None)
        if logger is None:
            log_file_name = f"trustworthy_{agent_arch}_{llm_model_name}.log"
            logger = TrustworthyAgentLogger(
                log_file_name=log_file_name,
                FLAG_PRINT=True
            )
        kwargs['logger'] = logger
        # ---------------------------------------------------

        # ---------------- BaseAgent ----------------
        super().__init__(name=name, role=role, llm=llm, actions=actions, **kwargs)
        self.max_exec_steps = 10 # Set max steps for agent following BOLAA
        # ---------------------------------------------------
        
        self.score_last_only = score_last_only         # Optional: Whether to score only final Finish act
        
        # ---------------- Minimal TLM Setup ----------------
        self.tlm = TLM()
        # ---------------------------------------------------

        # ---------------- Save to CSV (Optional) ----------------
        if trust_score_file is None:
            trust_score_file = f"data/trustworthy_{agent_arch}_{llm_model_name}.csv"
        self.trust_score_file = trust_score_file
        try:
            with open(self.trust_score_file, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'task_id',
                    'step',
                    'question',
                    'answer',
                    'prompt',
                    'raw_action',
                    'finish_response',
                    'trust_score',
                    'f1',
                    'action_name',
                    'action_params'
                ])
        except FileExistsError:
            pass  # File already exists
        # ---------------------------------------------------

    def __next_act__(self, task: TaskPackage, action_chain: ActObsChainType) -> AgentAct:
        """Override __next_act__ to track and score LLM interactions.
        
        Args:
            task: The current task being executed
            action_chain: History of actions and observations
            
        Returns:
            AgentAct: The next action to take
        """
        # ---------------- BaseAgent ----------------
        action_prompt = self.prompt_gen.action_prompt(
            task=task,
            actions=self.actions,
            action_chain=action_chain,
        )
        self.logger.get_prompt(action_prompt)
        raw_action = self.llm_layer(action_prompt)
        if raw_action.startswith('Action: '):
            raw_action = raw_action[8:]
        self.logger.get_llm_output(raw_action)
        agent_act = self.__action_parser__(raw_action)
        # ---------------------------------------------------

        # ---------------- Minimal TLM Setup ----------------
        trust_score = None
        if not self.score_last_only or agent_act.name == FinishAct.action_name:
            trust_score = self.tlm.get_trustworthiness_score(action_prompt, raw_action)["trustworthiness_score"]
        # ---------------------------------------------------

        # ---------------- Get F1 score & Response (Optional) ----------------
        f1 = None
        response = None
        if agent_act.name == FinishAct.action_name:
            response = FinishAct(**agent_act.params)
            f1, _, _ = f1_score(response, task.ground_truth)
        
        # ---------------- Logging & Saving (Optional) ----------------
        if trust_score is not None:
            self.logger.log_action_trust(
                action=agent_act,
                trust_score=trust_score,
                agent_name=self.name,
                step_idx=len(action_chain)
            )

            self.__record_interaction__(
                task_id=task.task_id,
                step=len(action_chain),
                question=task.instruction,
                answer=task.ground_truth,
                prompt=action_prompt,
                raw_action=raw_action,
                finish_response=response,
                trust_score=trust_score,
                f1=f1,
                action_name=agent_act.name,
                action_params=agent_act.params
            )
        # ---------------------------------------------------
        return agent_act

    # ---------------- Save to CSV (Optional) ----------------
    def __record_interaction__(self, **kwargs):
        """Record an LLM interaction to the trust score CSV file.
        
        Args:
            **kwargs: Interaction details to record
        """
        try:
            with open(self.trust_score_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    kwargs.get('task_id', ''),
                    kwargs.get('step', 0),
                    kwargs.get('question', ''),
                    kwargs.get('answer', ''),
                    kwargs.get('prompt', ''),
                    kwargs.get('raw_action', ''),
                    kwargs.get('finish_response', None),
                    kwargs.get('trust_score', None),
                    kwargs.get('f1', None),
                    kwargs.get('action_name', ''),
                    str(kwargs.get('action_params', {}))
                ])
        except Exception as e:
            self.logger.error(f"Failed to record trust score: {str(e)}")