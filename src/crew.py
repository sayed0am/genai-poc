# src/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
# from tool import document_retrieval_tool
from tool import DocumentRetrievalTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai import LLM
document_retrieval_tool = DocumentRetrievalTool(result_as_answer=True)
# GEN_MODEL = LLM(model="ollama/qwen3:4b-instruct-2507-q8_0", timeout=300,
#     verbose=True,temperature=0.7, keep_alive="10m", max_tokens=2048, max_completion_tokens=1024, top_p=0.8)

OPENROUTER_API_KEY="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"
GEN_MODEL = LLM(
    model="openrouter/z-ai/glm-4.5",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

@CrewBase
class RagCrew():
    """RagCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def document_researcher(self) -> Agent:
        return Agent(
            llm=GEN_MODEL,  # Use default LLM from config
            config=self.agents_config['document_researcher'], # type: ignore[index]
            verbose=True,
            tools=[document_retrieval_tool],
            respect_context_window=True,
            max_iter=2,
            allow_delegation=False
        )

    @agent
    def answer_synthesizer(self) -> Agent:
        return Agent(
            llm=GEN_MODEL,  # Use default LLM from config
            config=self.agents_config['answer_synthesizer'], # type: ignore[index]
            verbose=True,
            respect_context_window=True,
            max_iter=3
        )

    @task
    def retrieve_document_chunks(self) -> Task:
        return Task(
            config=self.tasks_config['retrieve_document_chunks'], # type: ignore[index]
        )

    @task
    def answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['answer_task'], # type: ignore[index]
            markdown=True
            )

    @crew
    def crew(self) -> Crew:
        """Creates the RagCrew"""
        # Create tasks explicitly to handle dependencies
        research_task = self.retrieve_document_chunks()
        answer_task = self.answer_task()
        
        # Set context after task creation
        answer_task.context = [research_task]
        
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=[research_task, answer_task], # Explicitly pass tasks in order
            process=Process.sequential,
            verbose=True,

        )