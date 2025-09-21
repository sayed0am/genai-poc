# src/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from tool import DocumentRetrievalTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai import LLM
document_tool = DocumentRetrievalTool()
GEN_MODEL = LLM(model="ollama/qwen3:4b-instruct-2507-q8_0", temperature=0.7, timeout=160.0 , keep_alive="10m", max_tokens=2048, max_completion_tokens=1024)
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
            tools=[document_tool],
            max_iter=3
        )

    @agent
    def answer_synthesizer(self) -> Agent:
        return Agent(
            llm=GEN_MODEL,  # Use default LLM from config
            config=self.agents_config['answer_synthesizer'], # type: ignore[index]
            verbose=True,
            max_iter=3
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['answer_task'], # type: ignore[index]
            )

    @crew
    def crew(self) -> Crew:
        """Creates the RagCrew"""
        # Create tasks explicitly to handle dependencies
        research_task = self.research_task()
        answer_task = self.answer_task()
        
        # Set context after task creation
        answer_task.context = [research_task]
        
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=[research_task, answer_task], # Explicitly pass tasks in order
            process=Process.sequential,
            verbose=True,

        )