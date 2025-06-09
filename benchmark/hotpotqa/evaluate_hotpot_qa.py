import argparse
import json
import os
import re
import csv
import string
from utils import f1_score
from dotenv import load_dotenv

import joblib
import numpy as np
import requests
from tqdm import tqdm
from SearchActions import WikipediaSearch
from hotpotagents import WikiSearchAgent 


from agentlite.actions import BaseAction, FinishAct, ThinkAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.terminal_logger import AgentLogger



def download_file(url, filename):
    """
    Download a file from a URL and save it locally.
    """
    response = requests.get(url)
    response.raise_for_status()  # Check if the download was successful
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {filename}")


def load_hotpot_qa_data(level):
    """
    Load HotpotQA data for a given level. If data doesn't exist, download it.
    """
    file_path = f"./data/{level}.joblib"
    data_url = (
        f"https://github.com/salesforce/BOLAA/raw/main/hotpotqa_run/data/{level}.joblib"
    )

    if not os.path.exists(file_path):
        print(f"{level} data not found, downloading...")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        download_file(data_url, file_path)
    # joblib requires python 3.10 or higher
    return joblib.load(file_path)

def run_hotpot_qa_agent_one_complex_level(level="easy", llm_name="gpt-3.5-turbo-16k-0613", agent_arch="react", PROMPT_DEBUG_FLAG=False, num_examples=None):
    """
    Test the WikiSearchAgent with a single specified dataset level and LLM.
    Args:
        level: Dataset difficulty level ("easy", "medium", "hard")
        llm_name: Name of the language model to use
        agent_arch: Agent architecture type
        PROMPT_DEBUG_FLAG: Whether to enable prompt debugging
        num_examples: Number of examples to evaluate (default: None)
    Returns:
        tuple: (average_f1_score, accuracy) for the specified level
    """
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

    # build the search agent
    llm_config = LLMConfig({
        "llm_name": llm_name, 
        "temperature": 0.0,
        "api_key": openai_api_key
    })
    # running xlam 
    if llm_name in ["xlam", "xlam_v2"]:
        llm_config = LLMConfig(
            {
                "llm_name": llm_name, 
                "temperature": 0.0, 
                "base_url": "http://localhost:8000/v1",
                "api_key": "EMPTY"
            }
        )
    llm = get_llm_backend(llm_config)
    agent = WikiSearchAgent(llm=llm, agent_arch=agent_arch, PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
    
    # Initialize results file for this level
    results_file = f"data/{agent_arch}_{llm_name}_results_{level}.csv"
    with open(results_file, "w", newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Question", "Ground Truth", "Prediction", "F1 Score", "Running Accuracy", "Execution Chain"])
    
    hotpot_data = load_hotpot_qa_data(level)
    hotpot_data = hotpot_data.reset_index(drop=True)
    if num_examples is not None:
        hotpot_data = hotpot_data.head(num_examples)
    task_instructions = [
        (row["question"], row["answer"]) for _, row in hotpot_data.iterrows()
    ]
    f1_list, correct, results = [], 0, {}
    for test_task, answer in tqdm(task_instructions, desc=f"Processing {level} level"):
        test_task_pack = TaskPackage(instruction=test_task, ground_truth=answer)
        response = agent(test_task_pack)
        execution = agent.short_term_memory.get_action_chain(task=test_task_pack)
        f1, _, _ = f1_score(response, answer)
        f1_list.append(f1)
        correct += int(response == answer)
        results[test_task] = (response, answer)

        avg_f1 = np.mean(f1_list)
        acc = correct / len(task_instructions)
        
        # Create CSV row with proper escaping and quoting
        row = [test_task, answer, response, f"{f1:.4f}", f"{acc:.4f}", str(execution)]
        with open(results_file, "a", newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(row)
            
    return avg_f1, acc

def run_hotpot_qa_agent(level=None, llm_name="gpt-3.5-turbo-16k-0613", agent_arch="react", PROMPT_DEBUG_FLAG=False, num_examples=None):
    """
    Test the WikiSearchAgent on HotPotQA benchmark.
    Args:
        level: Dataset difficulty level ("easy", "medium", "hard") or None for all levels
        llm_name: Name of the language model to use
        agent_arch: Agent architecture type
        PROMPT_DEBUG_FLAG: Whether to enable prompt debugging
        num_examples: Number of examples to evaluate per level (default: None)
    Returns:
        dict: Results for each level containing (f1_score, accuracy) tuples
    """
    if level is not None:
        if level not in ["easy", "medium", "hard"]:
            raise ValueError("Level must be one of: easy, medium, hard")
        f1, acc = run_hotpot_qa_agent_one_complex_level(
            level=level,
            llm_name=llm_name,
            agent_arch=agent_arch,
            PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG,
            num_examples=num_examples
        )
        return {level: (f1, acc)}
    else:
        results = {}
        for lvl in ["easy", "medium", "hard"]:
            print(f"\nRunning evaluation for {lvl} level...")
            f1, acc = run_hotpot_qa_agent_one_complex_level(
                level=lvl,
                llm_name=llm_name,
                agent_arch=agent_arch,
                PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG,
                num_examples=num_examples
            )
            results[lvl] = (f1, acc)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Search Agent on the HotPotQA Benchmark"
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["easy", "medium", "hard"],
        default=None,
        help="Difficulty level of the dataset. If not provided, runs all levels.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="gpt-3.5-turbo-16k-0613",
        help="Name of the language model",
    )
    parser.add_argument(
        "--agent_arch",
        type=str,
        choices=["react", "act", "planact", "planreact", "zs", "zst"],
        default="react",
        help="agent reasoning type",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="debug flag",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of examples to evaluate. If not provided, uses the entire dataset.",
    )
    args = parser.parse_args()

    results = run_hotpot_qa_agent(
        level=args.level, 
        llm_name=args.llm, 
        agent_arch=args.agent_arch, 
        PROMPT_DEBUG_FLAG=args.debug,
        num_examples=args.num_examples
    )
    
    print(f"{'+'*100}")
    for level, (f1, acc) in results.items():
        print(f"LLM model: {args.llm}, Dataset: {level}, Result: F1-Score = {f1:.4f}, Accuracy = {acc:.4f}")
    print(f"{'+'*100}")
