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
from google.adk.tools import tool, ToolContext
from google.genai import types
from google.adk.code_executors.agent_engine_sandbox_code_executor import AgentEngineSandboxCodeExecutor

SANDBOX = os.environ.get("SANDBOX_RESOURCE_NAME", "projects/605897091243/locations/us-central1/reasoningEngines/3475832232818507776/sandboxEnvironments/2303281147120975872")

def get_calc_agent():
    return Agent(
        model="gemini-2.5-flash",
        name="Calculator_Agent",
        instruction="You are a calculator. You are given a math question and context. You MUST use your built-in python sandbox to write python to calculate the answer. Simply output a pure markdown code block inside your text response to trigger the execution: ```python\\nprint(1+1)\\n```.",
        code_executor=AgentEngineSandboxCodeExecutor(sandbox_resource_name=SANDBOX)
    )

@tool
async def calculate_with_python(tool_context: ToolContext, math_question_with_data: str) -> str:
    """Calculates answers using a Python sandbox. Input must contain the raw numbers and the question.
    Example: 'Q1=100, Q2=120. What is the difference?'
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    
    print(f"   [Tool] Calculator SubAgent analyzing: {math_question_with_data}")
    
    agent = get_calc_agent()
    service = InMemorySessionService()
    await service.create_session("calc_app", "sys", "calc_session")
    runner = Runner(app_name="calc_app", agent=agent, session_service=service)
    
    events = runner.run_async(
        user_id="sys",
        session_id="calc_session",
        new_message=types.Content(role="user", parts=[types.Part.from_text(text=math_question_with_data)])
    )
    
    resp = ""
    async for event in events:
        e_type = getattr(event, "type", None) or getattr(event, "event_type", None)
        if e_type == "TEXT_MESSAGE_CONTENT":
            resp += getattr(event, "text", "")
        elif e_type == "RUN_CODE_CONTENT":
            for p in event.content.parts:
                out = getattr(p, "code_execution_result", None)
                if out:
                    resp += f"\n[Code Result]: {out.output}"
    return resp

async def main():
    ctx = ToolContext(session=None, agent=None)
    out = await calculate_with_python(ctx, "Q1=100, Q2=120. What is difference?")
    print("OUTPUT IS:", out)

if __name__ == "__main__":
    asyncio.run(main())
