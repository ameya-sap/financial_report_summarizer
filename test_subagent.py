import os
import asyncio
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "financial_supervisor", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get("GOOGLE_CLOUD_PROJECT", "gcpsaptesting")
os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.code_executors.agent_engine_sandbox_code_executor import AgentEngineSandboxCodeExecutor
from google.genai import types
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService

async def main():
    math_question_with_data = "Q1 2025 revenue is 90234 and Q1 2024 revenue is 80539. What is the difference between Q1 2025 revenue and Q1 2024 revenue?"
    SANDBOX_RESOURCE_NAME = os.environ.get("SANDBOX_RESOURCE_NAME", "projects/605897091243/locations/us-central1/reasoningEngines/3475832232818507776/sandboxEnvironments/2303281147120975872")
    
    print("Initializing Calc Agent...")
    calc_agent = Agent(
        model="gemini-2.5-flash",
        name="Calculator_Agent",
        instruction="""You are a strict code executor. You are given a math question containing raw data. 
You MUST use your built-in python sandbox to write python to calculate the answer.
Output a markdown block with the python code. Ensure you print() the final output.""",
        code_executor=AgentEngineSandboxCodeExecutor(sandbox_resource_name=SANDBOX_RESOURCE_NAME)
    )
    
    service = InMemorySessionService()
    await service.create_session(app_name="calc_app", user_id="sys", session_id="calc_sess")
    artifact_service = InMemoryArtifactService()
    runner = Runner(app_name="calc_app", agent=calc_agent, session_service=service, artifact_service=artifact_service)
    
    print("Running runner...")
    events = runner.run_async(
        user_id="sys",
        session_id="calc_sess",
        new_message=types.Content(role="user", parts=[types.Part.from_text(text=math_question_with_data)])
    )
    
    print("Processing events...")
    final_output = ""
    async for event in events:
        e_type = getattr(event, "type", None) or getattr(event, "event_type", None)
        print(f"EVENT > {e_type}")
        if e_type == "TEXT_MESSAGE_CONTENT":
            txt = getattr(event, "text", "")
            print(f"  TEXT: {txt}")
            final_output += txt + "\n"
        elif e_type == "RUN_CODE_CONTENT":
            for p in event.content.parts:
                out = getattr(p, "code_execution_result", None)
                if out:
                    print(f"  CODE RESULT: {out.output}")
                    final_output += f"\n[Code Result]: {out.output}\n"
    
    print("\n--- FINAL OUTPUT ---\n")
    print(final_output.strip())

if __name__ == "__main__":
    asyncio.run(main())
