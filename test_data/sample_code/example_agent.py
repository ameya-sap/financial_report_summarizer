import asyncio
import os
from pathlib import Path
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.code_executors.agent_engine_sandbox_code_executor import AgentEngineSandboxCodeExecutor
from google.genai import types
import vertexai

# Set environment variables for google-genai Client
os.environ["GOOGLE_CLOUD_PROJECT"] = "gcpsaptesting"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"

# Initialize Vertex AI (still good for other Vertex usage)
vertexai.init(project="gcpsaptesting", location="us-central1")

# Define the image path
IMAGE_PATH = Path("asset/hand.png")

# TODO: Replace with your actual Sandbox Resource Name
# Format: projects/{PROJECT_ID}/locations/{LOCATION_ID}/reasoningEngines/{REASONING_ENGINE_ID}/sandboxEnvironments/{SANDBOX_ID}
#
# To create a new Sandbox or list existing ones, run:
#   python sandbox_manager.py create --project_id YOUR_PROJECT_ID
#   python sandbox_manager.py list --project_id YOUR_PROJECT_ID
SANDBOX_RESOURCE_NAME = os.environ.get(
    "adk-code-exec-sandbox", 
    "projects/605897091243/locations/us-central1/reasoningEngines/3475832232818507776/sandboxEnvironments/2303281147120975872"
)

# Define the agent
agent = Agent(
    name="vision_agent",
    model= "gemini-2.5-flash", #"gemini-3.0-flash-preview",
    instruction=(
        "You are a helpful assistant with code execution capabilities. "
        #"You are a helpful assistant with vision and code execution capabilities. "
        #"When asked to analyze an image, you can use Python code to help with counting, "
        #"verification, or other logic. Annotate the image to help you with the reasonining. "
        #"You can also rely on your internal vision capabilities."
    ),
    code_executor=AgentEngineSandboxCodeExecutor(
        sandbox_resource_name=SANDBOX_RESOURCE_NAME
    )
)

async def main():
    if not IMAGE_PATH.exists():
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    print(f"Analyzing {IMAGE_PATH}...")
    
    # Read image bytes
    image_bytes = IMAGE_PATH.read_bytes()

    session_service = InMemorySessionService()
    await session_service.create_session(app_name="vision_agent_app", user_id="user_1", session_id="session_1")
    
    artifact_service = InMemoryArtifactService()
    runner = Runner(
        agent=agent, 
        app_name="vision_agent_app", 
        session_service=session_service,
        artifact_service=artifact_service
    )
    
    # Create the user message with image and text
    user_message = types.Content(
        role="user",
        parts=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part(text="Calculate the sum of the first 500 prime numbers. Write code for the calculation"),
            #types.Part(text="Tell me how many fingers am I holding?")
            #types.Part(text="Zoom into the expression pedals and tell me how many pedals are there?")
        ]
    )

    # Run the agent
    event_stream = runner.run_async(
        user_id="user_1",
        session_id="session_1",
        new_message=user_message
    )

    print("\n--- Agent Response ---")
    async for event in event_stream:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent (Text): {part.text}")
                if part.executable_code:
                    print(f"Agent (Code):\n{part.executable_code.code}")
                if part.code_execution_result:
                    print(f"Execution Result:\n{part.code_execution_result.output}")

if __name__ == "__main__":
    asyncio.run(main())
