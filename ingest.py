import os
import re
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.labels import DocItemLabel
from google.genai import Client
from google.genai import types
import chromadb
import uuid
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

# Configuration
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "gcpsaptesting")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_GENAI_USE_VERTEXAI = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", True)
EARNINGS_DIR = Path("earnings")
IMAGE_CACHE_DIR = EARNINGS_DIR / "image_cache"

# Ensure image cache directory exists
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize GenAI Client
client = Client(vertexai=bool(GOOGLE_GENAI_USE_VERTEXAI), project=GCP_PROJECT, location=GCP_LOCATION)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Clear the old collection to prevent duplicates with wrong metadata
try:
    chroma_client.delete_collection("financial_reports")
except Exception as e:
    print(f"Skipping deletion: {e}")

collection = chroma_client.get_or_create_collection(name="financial_reports")

# Configure Docling
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = 2.0
pipeline_options.generate_picture_images = True
pipeline_options.generate_page_images = False

doc_converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF],
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def describe_image(image_bytes: bytes) -> str:
    """Uses Gemini to describe a chart."""
    prompt = (
        "Describe this financial chart in detail. Extract all axes labels, "
        "key data points, trends, and the title. Format as Markdown."
    )
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
            prompt
        ]
    )
    return response.text

def process_document(pdf_path: Path):
    print(f"\nProcessing: {pdf_path}")
    
    # Extract structural metadata
    quarter = pdf_path.parent.name
    filename = pdf_path.name
    doc_type = "earnings-release" if "release" in filename.lower() else "earnings-slides"
    company = "alphabet" if "alphabet" in filename.lower() else "unknown"
    
    result = doc_converter.convert(str(pdf_path))
    doc = result.document
    
    markdown_lines = []
    current_heading = "Document Start"
    
    print("  Extracting elements and contextualizing images...")
    for element, level in doc.iterate_items():
        # Track Header Context
        if element.label == DocItemLabel.SECTION_HEADER or (hasattr(element.label, 'name') and element.label.name.startswith('heading')):
            if hasattr(element, "text") and element.text:
                current_heading = element.text
                
        if element.label == DocItemLabel.PICTURE:
            image_obj = element.get_image(doc)
            if image_obj:
                img_path = IMAGE_CACHE_DIR / quarter / f"{uuid.uuid4().hex}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                image_obj.save(img_path)
                
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                
                print(f"    -> Found chart under heading '{current_heading}', fetching Gemini description...")
                try:
                    description = describe_image(img_bytes)
                    
                    # Store chart as a standalone chunk in ChromaDB
                    chart_id = f"{filename}_chart_{uuid.uuid4().hex}"
                    chart_meta = {
                        "Quarter": quarter,
                        "Document_Type": "chart", # Explicitly label as a chart
                        "Company": company,
                        "Image_Path": str(img_path),
                        "Header_Path": current_heading,
                        "Chart_Type": "Financial Visual"
                    }
                    
                    chart_emb_res = client.models.embed_content(
                        model="text-embedding-005",
                        contents=description
                    )
                    
                    collection.add(
                        embeddings=[chart_emb_res.embeddings[0].values],
                        documents=[description],
                        metadatas=[chart_meta],
                        ids=[chart_id]
                    )
                    print(f"    -> Stored chart as atomic chunk.")
                except Exception as e:
                    print(f"    -> Failed to describe image: {e}")
            continue
            
        # Specifically parse tables as HTML, others as Markdown
        if element.label == DocItemLabel.TABLE:
            try:
                table_html = element.export_to_html(doc=doc)
                if table_html:
                    markdown_lines.append(f"```html\n{table_html}\n```")
            except Exception as e:
                print(f"    -> Warning: Could not export table to HTML: {e}")
        elif hasattr(element, "export_to_markdown"):
            md_text = element.export_to_markdown(doc=doc)
            if md_text:
                markdown_lines.append(md_text)

    full_markdown = "\n\n".join(markdown_lines)
    
    print("  Chunking Markdown textual semantics...")
    
    # Split by markdown headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    md_header_splits = markdown_splitter.split_text(full_markdown)
    
    # Sub-split large chunks (recursive character)
    chunk_size = 2000
    chunk_overlap = 500
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(md_header_splits)
    
    texts = []
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(splits):
        chunk_text = chunk.page_content
        
        # Build Header Path
        header_path = " > ".join(
            [chunk.metadata.get(f"Header {k}", "") for k in range(1, 4) if chunk.metadata.get(f"Header {k}")]
        )
        if not header_path:
             header_path = "Document Start"
        
        meta = {
            "Quarter": quarter,
            "Document_Type": doc_type,
            "Company": company,
            "Header_Path": header_path
        }
            
        texts.append(chunk_text)
        metadatas.append(meta)
        ids.append(f"{filename}_text_chunk_{i}")

    print(f"  Generated {len(texts)} text chunks. Vectorizing...")
    
    if texts:
        embeddings = []
        for text in texts:
            emb_res = client.models.embed_content(
                model="text-embedding-005",
                contents=text
            )
            embeddings.append(emb_res.embeddings[0].values)
            
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print("  Successfully stored in ChromaDB.")

def main():
    pdf_files = list(EARNINGS_DIR.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs.")
    for pdf_path in pdf_files:
        try:
            process_document(pdf_path)
        except Exception as e:
            print(f"Failed to process {pdf_path}: {e}")

if __name__ == "__main__":
    main()
