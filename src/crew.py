from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from tool import DocumentRetrievalTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai import LLM
import os

# Initialize document retrieval tool
document_retrieval_tool = DocumentRetrievalTool(result_as_answer=True)
DOCKER_ENV = os.getenv('DOCKER_ENV', False)
if DOCKER_ENV:
    uri = "host.docker.internal"
else:
    uri = "localhost"
GEN_MODEL = LLM(api_base=f"http://{uri}:11434", model="ollama/phi4-mini:3.8b-q8_0", timeout=300,
    verbose=True,temperature=0.0, keep_alive="10m", max_tokens=4096, max_completion_tokens=1024)

# GEN_MODEL = LLM(
#     model="openrouter/google/gemini-2.5-flash",
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv('OPENROUTER_API_KEY'),
#     temperature=0.3,
#     max_tokens=4096
# )


@CrewBase
class RagCrew():
    """Efficient RAG Crew with conversation history support"""

    agents: List[BaseAgent]
    tasks: List[Task]


    @agent
    def document_researcher(self) -> Agent:
        return Agent(
            llm=GEN_MODEL,
            config=self.agents_config['document_researcher'], # type: ignore[index]
            verbose=True,
            tools=[document_retrieval_tool],
            max_iter=2,
            allow_delegation=False
        )

    @agent
    def answer_synthesizer(self) -> Agent:
        return Agent(
            llm=GEN_MODEL,
            config=self.agents_config['answer_synthesizer'], # type: ignore[index]
            verbose=True,
            max_iter=2,
            allow_delegation=False
        )

    @task
    def retrieve_document_chunks(self) -> Task:
        return Task(
            config=self.tasks_config['retrieve_document_chunks'], # type: ignore[index]
        )

    @task
    def synthesize_answer(self) -> Task:
        return Task(
            config=self.tasks_config['synthesize_answer'], # type: ignore[index]
            markdown=True
        )

    @crew
    def crew(self) -> Crew:
        """Creates the optimized RAG Crew"""
        research_task = self.retrieve_document_chunks()
        synthesis_task = self.synthesize_answer()

        # Set task dependency
        synthesis_task.context = [research_task]

        return Crew(
            agents=self.agents,
            tasks=[research_task, synthesis_task],
            process=Process.sequential,
            verbose=True,
            max_rpm=10  # Rate limiting
        )

