import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from dotenv import load_dotenv
from crew import RagCrew

# --- Arize Phoenix Tracing Setup ---
# This block configures the tracer to send data to your local Phoenix instance.
# It should be at the very top of your application's entry point.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DOCKER_ENV = os.getenv('DOCKER_ENV', False)
if DOCKER_ENV:
    uri = "host.docker.internal"
else:
    uri = "localhost"

phoenix_host = os.getenv("PHOENIX_HOST", uri)
# phoenix_host = os.getenv("PHOENIX_HOST", "localhost")  # Use localhost if not in Docker

phoenix_endpoint = f"http://{phoenix_host}:6006"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint

try:
    from phoenix.otel import register
    tracer_provider = register(
        project_name="default",
        endpoint=f"{phoenix_endpoint}/v1/traces",
        auto_instrument=True  # This automatically instruments CrewAI and other libraries
    )
    logging.info(f"✅ Arize Phoenix tracing successfully initialized for API server at {phoenix_endpoint}")
except ImportError as e:
    logging.warning(f"⚠️  Phoenix module not found: {e}. Install with: pip install arize-phoenix")
except Exception as e:
    logging.warning(f"⚠️  Could not initialize Arize Phoenix tracing: {e}")
# --- End of Tracing Setup ---

# Ensure the project root is in the Python path
import sys
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import your existing crew creation function
# from src.rag_system.crew import create_rag_crew

# Load environment variables
load_dotenv()

# Initialize the FastAPI app
app = FastAPI(
    title="CrewAI RAG API",
    description="An API server for the agentic RAG pipeline.",
    version="1.0.0",
)

# Add CORS middleware to allow requests from OpenWebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define the request model to be compatible with OpenAI's format
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]

@app.get("/v1/models")
def list_models():
    """
    OpenAI-compatible endpoint to list available models.
    This is required for OpenWebUI to discover available models.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "policy-crew",
                "object": "model", 
                "created": 1758516814,
                "owned_by": "policy-crew",
                "permission": [],
                "root": "policy-crew",
                "parent": None,
                "max_tokens": 262144,
                "context_length": 262144     
            }
        ]
    }

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible endpoint to interact with the CrewAI RAG pipeline.
    """
    # Extract the last user message as the query
    user_message = next((msg["content"] for msg in reversed(request.messages) if msg["role"] == "user"), None)

    # Get last 10 user messages for conversation history
    user_messages = [msg["content"] for msg in request.messages[-10:] if msg["role"] == "user"]
    logging.info(f"User messages: {user_messages}")

    if not user_message:
        return {"error": "No user message found"}

    print(f"Received query for API: {user_message}")

    # Create conversation history context from user messages
    conversation_history = "\n".join([f"Q: {msg}" for msg in user_messages[:-1]])  # Exclude current query

    # Create inputs with conversation history
    inputs = {
        'query': user_message,
        'conversation_history': conversation_history if conversation_history else "No previous conversation"
    }

    # Initialize crew and run
    rag_crew = RagCrew()
    result = rag_crew.crew().kickoff(inputs=inputs)
    # Format the response to be compatible with the OpenAI API standard
    response = {
        "id": "policyl-123", # Dummy ID
        "object": "chat.completion",
        "created": 1758516814, # Dummy timestamp
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": str(result), # Ensure the result is a string
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0, # You can implement token counting if needed
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    return response

if __name__ == "__main__":
    # This allows you to run the API server directly for testing
    uvicorn.run(app, host="0.0.0.0", port=8000)
