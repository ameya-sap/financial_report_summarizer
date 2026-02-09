import os
import chromadb
from google.adk.agents import LlmAgent
from google.genai import Client

from dotenv import load_dotenv

load_dotenv()

# Configuration
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "please_update")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "1")

# Initialize GenAI Client
client = Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

# Initialize ChromaDB
# Initialize ChromaDB using absolute path so adk web finds it
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="financial_reports")

def retrieve_financial_data(query: str, quarter: str = None) -> str:
    """Retrieves exact financial details, tables, or chart descriptions from reports.
    
    Args:
        query: The user's question about the financial data.
        quarter: Optional quarter (e.g., 'Q1-2025') to filter the research. Use this if the user specifies a quarter.
    """
    where_clause = {}
    if quarter:
        where_clause["Quarter"] = quarter
        
    # Generate embedding for the query using the same model as ingestion
    emb_res = client.models.embed_content(
        model="text-embedding-005",
        contents=query
    )
    query_embedding = emb_res.embeddings[0].values
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=15,
        where=where_clause if where_clause else None
    )
    
    # Format results
    if not results['documents'][0]:
        return "No relevant financial data found for this query."
        
    formatted_results = []
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        
        chunk_text = f"--- Result {i+1} ---\n"
        chunk_text += f"Source: {meta.get('Company', 'Unknown')} {meta.get('Quarter', '')} {meta.get('Document_Type', '')}\n"
        chunk_text += f"Section: {meta.get('Header_Path', '')}\n"
        
        # Inject explicit image link if present
        if meta.get("Image_Path"):
            chunk_text += f"[Source Image: {meta['Image_Path']}]\n"
            
        chunk_text += f"\n{doc}\n"
        formatted_results.append(chunk_text)
        
    return "\n".join(formatted_results)

# Create the Google ADK Python Agent
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="Financial_Analyst",
    instruction=(
        "You are a financial analyst. Read the exact details from the provided tool "
        "to answer the user's questions about earnings. "
        "When using the retrieve_financial_data tool, DO NOT pass the user's full conversational sentence. "
        "Extract only the core financial keywords. "
        "If the initial tool call does not contain the answer, you must NOT give up. Call the tool a second time using synonymous financial terms (e.g., if 'dividend' fails, try 'stockholder return'). "
        "Answer the user's queries directly based on the text. If a user asks about an entity that the document groups together (e.g., 'Class A, Class B, and Class C shares'), provide the aggregate figure present in the text. Explicitly state that the figure is a combined aggregate rather than apologizing for a missing breakdown."
        "When users ask for a specific segment like 'Cloud', "
        "verify that the retrieved 'Section' or 'Chart Description' explicitly mentions 'Cloud'. "
        "If you find multiple charts (e.g., Services vs Cloud), only display the one that matches the LOB. "
        "CRITICAL: If the retrieved context includes a [Source Image: <path>] tag "
        "or if the Document_Type is 'chart' and an Image_Path is present, "
        "you MUST include a markdown image link like `![Chart Description](<path>)` "
        "in your final answer so the user can see the chart."
    ),
    tools=[retrieve_financial_data],
)

if __name__ == "__main__":
    import asyncio
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    async def run_cli():
        runner = Runner(app_name="Financial_App", agent=root_agent, session_service=InMemorySessionService())
        print("Financial Analyst Agent (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            events = runner.run_async(
                user_id="cli_user",
                session_id="cli_session",
                new_message=types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
            )
            
            print("\nAnalyst: ", end="", flush=True)
            async for event in events:
                if event.type == "TEXT_MESSAGE_CONTENT":
                    print(event.text, end="", flush=True)
            print("\n" + "-" * 50)

    asyncio.run(run_cli())
