import time
import uuid

from pydantic import BaseModel, Field


class TaskPackage(BaseModel):
    instruction: str
    completion: str = "active"
    creator: str = ""
    timestamp: str = time.time()
    answer: str = ""
    ground_truth: str = ""
    executor: str = ""
    priority: int = 5
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def __str__(self):
        return f"""Task ID: {self.task_id}\nInstruction: {self.instruction}\nTask Creator: {self.creator}\nTask Completion:{self.completion}\nAnswer: {self.answer}\nTask Executor: {self.executor}"""
