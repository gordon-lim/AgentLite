import re
import numpy as np
from textwrap import dedent

CONFIDENCE_PROMPT = """Evaluate how confident you are that the given Answer is a good and accurate response to the Question.
Please assign a Score using the following 5-point scale:
1: You are not confident that the Answer addresses the Question at all, the Answer may be entirely off-topic or irrelevant to the Question.
2: You have low confidence that the Answer addresses the Question, there are doubts and uncertainties about the accuracy of the Answer.
3: You have moderate confidence that the Answer addresses the Question, the Answer seems reasonably accurate and on-topic, but with room for improvement.
4: You have high confidence that the Answer addresses the Question, the Answer provides accurate information that addresses most of the Question.
5: You are extremely confident that the Answer addresses the Question, the Answer is highly accurate, relevant, and effectively addresses the Question in its entirety.
The output should strictly use the following template: Explanation: [provide a brief reasoning you used to derive the rating Score] and then write 'Score: <rating>' on the last line.
"""

regex_list = [
    re.compile(
        r".*(?:^|\n)\s*score:?\s*\(?(?P<answer>one|two|three|four|five|[12345])\)?",
        flags=re.DOTALL | re.IGNORECASE
    ),
    re.compile(
        r".*\(?(?P<answer>one|two|three|four|five|[12345])\)?",
        flags=re.DOTALL | re.IGNORECASE
    )
]

def construct_confidence_prompt(prompt, response):
    """Construct a confidence evaluation prompt for self-evaluation scoring.
    
    Args:
        prompt: The original prompt/question
        response: The response/answer to evaluate
        
    Returns:
        str: Formatted confidence evaluation prompt
    """
    template = f"""Question:
        {prompt}
        Answer: {response}
        {CONFIDENCE_PROMPT}"""
    return dedent(template)

def parse_self_eval_score(text: str):
    """Parse the self-evaluation score from the LLM response.
    
    Args:
        text: The LLM response containing the confidence score
        
    Returns:
        float: Normalized confidence score (0-1 range) or np.nan if parsing fails
    """
    try:
        score = int(text.split("Score:")[-1].strip())
        return (score - 1) / 4  # Normalize to 0-1 range
    except:
        return np.nan

def get_self_eval_score(llm, prompt, response):
    """Get self-evaluation confidence score using the provided LLM.
    
    Args:
        llm: The language model to use for self-evaluation
        prompt: The original prompt/question
        response: The response/answer to evaluate
        
    Returns:
        float: Normalized confidence score (0-1 range) or np.nan if evaluation fails
    """
    try:
        confidence_prompt = construct_confidence_prompt(prompt, response)
        eval_response = llm(confidence_prompt)
        return parse_self_eval_score(eval_response)
    except Exception as e:
        print(f"Error in self-evaluation: {e}")
        return np.nan 