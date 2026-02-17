import os
import asyncio
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get("GOOGLE_CLOUD_PROJECT", "gcpsaptesting")
os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .prompt import SUPERVISOR_INSTRUCTION
from .tools import retrieve_narrative, retrieve_financial_tables, calculate_with_python

root_agent = Agent(
    model="gemini-2.5-flash",
    name="Financial_Supervisor",
    instruction=SUPERVISOR_INSTRUCTION,
    tools=[retrieve_narrative, retrieve_financial_tables, calculate_with_python]
)
