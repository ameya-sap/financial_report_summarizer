import sys, os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get("GOOGLE_CLOUD_PROJECT", "gcpsaptesting")
os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

sys.path.insert(0, os.path.dirname(__file__))

import asyncio
from typing import cast
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.planners.base_planner import BasePlanner
from financial_report_summarizer.agent import root_agent

async def main():
    llm = getattr(root_agent, "llm", None) or getattr(root_agent, "_llm", None)
    if llm:
        print(f"LLM: {llm}")
    print("Tools attached to LlmAgent:", getattr(root_agent, "tools", None) or getattr(root_agent, "_tools", None))
    print("Code Executor attached to LlmAgent:", getattr(root_agent, "code_executor", None))
    
    # Try to simulate an LlmRequest to see the tools dict
    from google.adk.flows.llm_flows.single_flow import SingleFlow
    print("Agent type:", type(root_agent))

if __name__ == "__main__":
    asyncio.run(main())
