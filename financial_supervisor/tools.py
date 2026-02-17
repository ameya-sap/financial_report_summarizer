import os
import mimetypes
from google.adk.tools import ToolContext
import google.genai.types as types
from google.genai import Client
import chromadb
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "gcpsaptesting")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
client = Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

# Initialize ChromaDB using absolute path so adk web finds it
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="financial_reports")

async def retrieve_narrative(tool_context: ToolContext, query: str, quarter: str = "") -> str:
    """
    Retrieves TEXT narratives, executive quotes, risk factors, and strategic commentary.
    Use this for questions like "What did the CEO say?", "What are the headwinds?", "Summarize the outlook".
    This tool DOES NOT return financial tables.
    IMPORTANT: The 'quarter' argument must exactly match the document folder (e.g., 'Q1-2025'). 
    If you are looking for historical comparisons (like Q1 2024), leave 'quarter' blank, as historical data is usually contained in the current quarter's report.
    """
    print(f"   [Tool] Searching Narrative for: {query}")
    if quarter:
        where_clause = {
            "$and": [
                {"Content_Type": {"$eq": "text"}},
                {"Quarter": {"$eq": quarter}}
            ]
        }
    else:
        where_clause = {"Content_Type": "text"}
        
    emb_res = client.models.embed_content(model="text-embedding-005", contents=query)
    
    results = collection.query(
        query_embeddings=[emb_res.embeddings[0].values],
        n_results=10, 
        where=where_clause 
    )

    if not results['documents'][0]:
        return "No relevant narrative text found."

    formatted = []
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        formatted.append(f"--- TEXT FROM SECTION: {meta.get('Header_Path', 'Unknown')} ---\n{doc}")
    
    return "\n\n".join(formatted)

async def retrieve_financial_tables(tool_context: ToolContext, query: str, quarter: str = "") -> str:
    """
    Retrieves HTML/Markdown TABLES and CHARTS containing raw financial numbers.
    Use this for questions like "What was the revenue?", "Operating margin", "Balance sheet data".
    This tool is best for extracting raw data before performing calculations.
    IMPORTANT: The 'quarter' argument must exactly match the document folder (e.g., 'Q1-2025'). 
    If you are looking for historical comparisons (like Q1 2024), leave 'quarter' blank, as historical data is usually contained in the current quarter's report.
    """
    print(f"   [Tool] Searching Tables/Charts for: {query}")
    
    where_clause = {
        "$or": [
            {"Content_Type": {"$eq": "table"}},
            {"Content_Type": {"$eq": "chart"}}
        ]
    }
    
    if quarter:
        where_clause = {
            "$and": [
                {"Quarter": {"$eq": quarter}},
                {"$or": [
                    {"Content_Type": {"$eq": "table"}},
                    {"Content_Type": {"$eq": "chart"}}
                ]}
            ]
        }

    emb_res = client.models.embed_content(model="text-embedding-005", contents=query)
    
    results = collection.query(
        query_embeddings=[emb_res.embeddings[0].values],
        n_results=5, 
        where=where_clause 
    )

    if not results['documents'][0]:
        return "No relevant financial tables or charts found."

    formatted_results = []
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        
        chunk_text = f"--- DATA FROM SECTION: {meta.get('Header_Path', 'Unknown')} ---\n"
        
        # Inject explicit image link if present
        if meta.get("Image_Path"):
            img_path = meta['Image_Path']
            abs_img_path = img_path if os.path.isabs(img_path) else os.path.join(BASE_DIR, img_path)
            try:
                mime_type, _ = mimetypes.guess_type(abs_img_path)
                if not mime_type:
                    mime_type = "image/png"
                with open(abs_img_path, "rb") as bf:
                    image_bytes = bf.read()
                image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                artifact_id = await tool_context.save_artifact(image_part, os.path.basename(img_path))
                chunk_text += f"[Source Image Artifact: '{artifact_id}']\n"
            except Exception as e:
                chunk_text += f"[Source Image: {img_path}]\n"
                
        chunk_text += f"{doc}\n"
        formatted_results.append(chunk_text)
        
    return "\n\n".join(formatted_results)

async def calculate_with_python(tool_context: ToolContext, math_question_with_data: str) -> str:
    """Takes a math question along with raw numbers, and writes/executes a python script to answer it.
    Example math_question_with_data: "Q1 revenue is 100, Q2 is 120. What is the growth percentage?"
    """
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.code_executors.agent_engine_sandbox_code_executor import AgentEngineSandboxCodeExecutor
    
    from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
    
    print(f"   [Tool] Passing to Calculator SubAgent: {math_question_with_data}")
    
    SANDBOX_RESOURCE_NAME = os.environ.get("SANDBOX_RESOURCE_NAME", "projects/605897091243/locations/us-central1/reasoningEngines/3475832232818507776/sandboxEnvironments/2303281147120975872")
    
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
    
    events = runner.run_async(
        user_id="sys",
        session_id="calc_sess",
        new_message=types.Content(role="user", parts=[types.Part.from_text(text=math_question_with_data)])
    )
    
    final_output = ""
    async for event in events:
        e_type = getattr(event, "type", None) or getattr(event, "event_type", None)
        if e_type == "TEXT_MESSAGE_CONTENT":
            final_output += getattr(event, "text", "") + "\n"
        elif e_type == "RUN_CODE_CONTENT":
            for p in event.content.parts:
                out = getattr(p, "code_execution_result", None)
                if out:
                    final_output += f"\n[Code Result]: {out.output}\n"
                    
    return final_output.strip()
