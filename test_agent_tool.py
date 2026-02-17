import sys, os, asyncio
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get("GOOGLE_CLOUD_PROJECT", "gcpsaptesting")
os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
sys.path.insert(0, os.path.dirname(__file__))

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.code_executors.agent_engine_sandbox_code_executor import AgentEngineSandboxCodeExecutor

SANDBOX_RESOURCE_NAME = os.environ.get("SANDBOX_RESOURCE_NAME", "projects/605897091243/locations/us-central1/reasoningEngines/3475832232818507776/sandboxEnvironments/2303281147120975872")

code_agent = Agent(
    model="gemini-2.5-flash",
    name="Calculator_Agent",
    instruction="You are a strict code executor. You are given a math question and context. You MUST use your built-in code executor to write python. Always print the final output.",
    code_executor=AgentEngineSandboxCodeExecutor(sandbox_resource_name=SANDBOX_RESOURCE_NAME)
)

calc_tool = AgentTool(agent=code_agent)

supervisor = Agent(
    model="gemini-2.5-flash",
    name="Financial_Supervisor",
    instruction="Ask the calculator agent a simple math problem: what is 90234 - 80539?",
    tools=[calc_tool]
)

async def main():
    service = InMemorySessionService()
    runner = Runner(app_name="TestApp", agent=supervisor, session_service=service)
    await service.create_session(app_name="TestApp", user_id="test_user", session_id="test_session")
    
    events = runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="Calculate difference.")])
    )
    
    async for event in events:
        e_type = getattr(event, "type", None) or getattr(event, "event_type", None)
        if not e_type: continue
        if e_type == "TEXT_MESSAGE_CONTENT": print(getattr(event, 'text', ''))
        if e_type == "TOOL_CALL_CONTENT": print("[Tool Call]")

if __name__ == "__main__":
    asyncio.run(main())
