import sys
import os

# Insert the base directory into sys.path so 'financial_supervisor' can be imported by ADK web
sys.path.insert(0, os.path.dirname(__file__))

from financial_supervisor.agent import root_agent
