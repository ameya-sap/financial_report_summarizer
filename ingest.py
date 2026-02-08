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
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "please_update")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_GENAI_USE_VERTEXAI = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", True)
EARNINGS_DIR = Path("earnings")
IMAGE_CACHE_DIR = EARNINGS_DIR / "image_cache"

# Ensure image cache directory exists
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize GenAI Client
client = Client(vertexai=GOOGLE_GENAI_USE_VERTEXAI, project=GCP_PROJECT, location=GCP_LOCATION)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
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
    
    print("  Extracting elements and contextualizing images...")
    for element, level in doc.iterate_items():
        if element.label == DocItemLabel.PICTURE:
            image_obj = element.get_image(doc)
            if image_obj:
                img_path = IMAGE_CACHE_DIR / quarter / f"{uuid.uuid4().hex}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                image_obj.save(img_path)
                
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                
                print(f"    -> Found chart, fetching Gemini description...")
                try:
                    description = describe_image(img_bytes)
                    # Insert the text description and metadata link directly into the markdown flow
                    markdown_lines.append(f"\n[Chart Description]\n{description}\n[Source Image: {img_path}]\n")
                except Exception as e:
                    print(f"    -> Failed to describe image: {e}")
            continue
            
        # For tables, text, headings, docling provides export_to_markdown
        if hasattr(element, "export_to_markdown"):
            md_text = element.export_to_markdown(doc=doc)
            if md_text:
                markdown_lines.append(md_text)

    full_markdown = "\n\n".join(markdown_lines)
    
    print("  Chunking Markdown semantics...")
    
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
    chunk_overlap = 200
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
        
        # Look for image path in the chunk text
        img_match = re.search(r"\[Source Image: (.*?)\]", chunk_text)
        if img_match:
            meta["Image_Path"] = img_match.group(1)
            
        texts.append(chunk_text)
        metadatas.append(meta)
        ids.append(f"{filename}_chunk_{i}")

    print(f"  Generated {len(texts)} chunks. Vectorizing with text-embedding-005...")
    
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
