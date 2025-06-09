from typing import List, Dict, Any
import csv
from datetime import datetime
from dotenv import load_dotenv
from .BaseAgent import BaseAgent
from agentlite.commons import TaskPackage, AgentAct
from agentlite.commons.AgentAct import ActObsChainType
from cleanlab_tlm import TLM
from agentlite.logging.terminal_logger import TrustworthyAgentLogger

# Load environment variables
load_dotenv()

class TrustworthyAgent(BaseAgent):
    """A BaseAgent that tracks and scores LLM interactions for trustworthiness.
    
    This agent extends BaseAgent to add trustworthiness scoring functionality.
    It tracks all LLM interactions (prompts and responses) and saves them to CSV.
    
    Additional parameters:
        trust_score_file: str, optional
            Path to save trustworthiness scores. Defaults to "trust_scores_{timestamp}.csv"
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        llm: Any,
        actions: List[Any] = [],
        trust_score_file: str = None,
        **kwargs
    ):
        # Get LLM model name and agent architecture for logging
        llm_model_name = getattr(llm, "llm_name", "unk")
        agent_arch = kwargs.get("agent_arch", "unk")
        
        # Initialize logger first so we can use it in the rest of initialization
        logger = kwargs.pop('logger', None)
        if logger is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_name = f"trustworthy_{agent_arch}_{llm_model_name}_{timestamp}.log"
            logger = TrustworthyAgentLogger(
                log_file_name=log_file_name,
                FLAG_PRINT=True
            )
        kwargs['logger'] = logger
        
        super().__init__(name=name, role=role, llm=llm, actions=actions, **kwargs)
        
        # Initialize the TLM model
        self.tlm = TLM()

        # Initialize trust score tracking
        self.trust_scores = []
        if trust_score_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trust_score_file = f"data/trustworthy_{agent_arch}_{llm_model_name}_{timestamp}.csv"
        self.trust_score_file = trust_score_file
        
        # Create CSV file with headers if it doesn't exist
        try:
            with open(self.trust_score_file, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'task_id',
                    'step',
                    'prompt',
                    'raw_action',
                    'trust_score',
                    'action_name',
                    'action_params'
                ])
        except FileExistsError:
            pass  # File already exists, that's fine

    def __next_act__(self, task: TaskPackage, action_chain: ActObsChainType) -> AgentAct:
        """Override __next_act__ to track and score LLM interactions.
        
        Args:
            task: The current task being executed
            action_chain: History of actions and observations
            
        Returns:
            AgentAct: The next action to take
        """
        # Get the prompt and generate action as normal
        action_prompt = self.prompt_gen.action_prompt(
            task=task,
            actions=self.actions,
            action_chain=action_chain,
        )
        self.logger.get_prompt(action_prompt)
        raw_action = self.llm_layer(action_prompt)
        self.logger.get_llm_output(raw_action)
        
        # Parse the action
        agent_act = self.__action_parser__(raw_action)
        
        # Calculate trust score
        trust_score = self.tlm.get_trustworthiness_score(action_prompt, raw_action)["trustworthiness_score"]
        
        # Log the action with trust score using TrustworthyAgentLogger
        self.logger.log_action_trust(
            action=agent_act,
            trust_score=trust_score,
            agent_name=self.name,
            step_idx=len(action_chain)
        )

        # Record the interaction
        self.__record_interaction__(
            task_id=task.task_id,
            step=len(action_chain),
            prompt=action_prompt,
            raw_action=raw_action,
            trust_score=trust_score,
            action_name=agent_act.name,
            action_params=agent_act.params
        )
        
        # Update trust scores list
        self.trust_scores.append(trust_score)
        
        return agent_act

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
                    kwargs.get('prompt', ''),
                    kwargs.get('raw_action', ''),
                    kwargs.get('trust_score', 0.0),
                    kwargs.get('action_name', ''),
                    str(kwargs.get('action_params', {}))
                ])
        except Exception as e:
            self.logger.error(f"Failed to record trust score: {str(e)}")

    def get_observation(self, obs: str) -> str:
        """Override get_observation to include trust score logging."""
        # Calculate trust score for observation
        trust_score = self.tlm.get_trustworthiness_score("", obs)["trustworthiness_score"]
        
        # Log observation with trust score
        self.logger.log_observation_trust(obs, trust_score)
        
        return super().get_observation(obs)