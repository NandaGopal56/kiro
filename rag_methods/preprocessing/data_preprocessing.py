import os
import io
import logging
import warnings
import base64

from dotenv import load_dotenv
from collections import Counter

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from unstructured.documents.elements import Image, ListItem, NarrativeText, Table, Title
from unstructured.partition.pdf import partition_pdf

from .utils import clean_text, clean_chunk

load_dotenv()
print(f'API Key loaded: {bool(os.getenv("OPENAI_API_KEY"))}')

warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)

PDF_PATH = "../data/coffee_processing.pdf"

# Parse PDF into raw blocks
def parse_pdf(path):
    with open(path, "rb") as f:
        pdf_bytes = io.BytesIO(f.read())
    return partition_pdf(
        file=pdf_bytes,
        infer_table_structure=True,
        extract_images_in_pdf=True,
    )


def build_blocks(elements):
    """
    Convert unstructured elements into a flat list of dicts.
    """
    blocks = []

    for el in elements:
        page = el.metadata.page_number if el.metadata else None

        metadata = {
            "page"   : page,
        }

        if isinstance(el, (NarrativeText, ListItem)) and el.text:
            blocks.append({**metadata, "type": "text", "content": el.text})

        elif isinstance(el, Table):
            blocks.append({**metadata, "type": "table", "content": el.metadata.text_as_html})

        elif isinstance(el, Image):
            image_path = getattr(el.metadata, "image_path", None)

            blocks.append({
                **metadata,
                "type": "image",
                "content": el.text or "",
                "image_path": image_path,
            })

    return blocks



class ImageSummary(BaseModel):
    useful_image: bool
    description: str

class TableSummary(BaseModel):
    useful_table: bool
    description: str

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=400)
structured_llm = llm.with_structured_output(ImageSummary)
structured_table_llm = llm.with_structured_output(TableSummary)

# LLM summarization helpers 
def get_context_window(blocks, index, window=3):
    """
    Returns a string of text from the blocks immediately before and after
    the current index. Gives the LLM enough surrounding context to produce
    a meaningful summary without sending the entire document.
    Only text blocks are used — tables/images as neighbours add noise.
    """
    neighbours = [
        b["content"]
        for b in blocks[max(0, index - window): index + window + 1]
        if b["type"] == "text" and b["content"].strip()
    ]
    return "\n".join(neighbours)


def summarize_table(html, context):

    messages = [
            SystemMessage(
                content=(
                    "You analyze tables extracted from documents. Your goal is to produce a dense, "
                    "information-rich description suitable for semantic search and vector retrieval."
                )
            ),
            HumanMessage(
                content=f"""
                    Context around the table:
                    {context}

                    Table (HTML):
                    {html}

                    Tasks:

                    1. Determine whether the table contains meaningful structured information.
                    2. If the table is NOT meaningful (layout table, empty cells, repeated headers, decorative, etc.), return:
                    NOT_USEFUL

                    3. If the table IS meaningful, generate a dense description that captures the key information in a way
                    that will be useful for semantic search.

                    Your description should include:
                    - The topic or subject of the table.
                    - Important entities (products, locations, processes, categories, etc.).
                    - Key metrics, values, or measurements.
                    - Relationships or comparisons between rows/columns.
                    - Any notable trends, rankings, or differences.
                    - Units or scales if present.

                    Write 3 to 5 concise sentences. Avoid generic wording. Include concrete values, column meanings,
                    and important entities so the description is highly searchable.
                """
            ),
    ]

    result = structured_table_llm.invoke(messages)

    return result.model_dump()


def summarize_image(block, context):

    image_path = block.get("image_path")

    if not image_path:
        return {"useful_image": False, "description": "No image path provided."}

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        SystemMessage(
            content=(
                "You analyze images extracted from documents. "
                "Determine if the image contains meaningful information and produce a "
                "dense semantic description suitable for vector search."
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                },
                {
                    "type": "text",
                    "text": f"""
                        Context around the image:
                        {context}

                        Tasks:

                        1. Determine whether this image contains useful information related to the document.

                        2. If the image is NOT useful (logo, decoration, watermark, page border, blank image, repeated icon),
                        return:
                        NOT_USEFUL

                        3. If the image IS useful, produce a dense description optimized for semantic search.

                        Your description should capture:
                        - The topic or subject of the image
                        - Important entities, objects, or labels
                        - Any text visible in the image
                        - Data shown in charts/graphs/diagrams
                        - Relationships, trends, or processes illustrated
                        - Units, numbers, or measurements if present

                        Write 3 to 5 concise sentences with concrete details so the description is highly searchable.
                    """
                },
            ]
        ),
    ]

    result = structured_llm.invoke(messages)

    return result.model_dump()


def preprocess_data(pdf_path):
    elements = parse_pdf(pdf_path)
    blocks   = build_blocks(elements)

    type_counts = Counter(b["type"] for b in blocks)
    print(f"Total blocks: {len(blocks)} | {dict(type_counts)}")
    
    return blocks

def process_blocks(blocks):
    text_store = []
    table_store = []
    image_store = []

    for i, block in enumerate(blocks):
        btype = block["type"]
        page  = block.get("page")

        if btype == "text":
            # Collect text content sequentially — join later for chunking flexibility
            text_store.append({
                "content": block["content"],
                "type": "text",
                "page": page,
                "section": block.get("section"),
            })

        elif btype == "table":
            context = get_context_window(blocks, i)
            summary = summarize_table(block["content"], context)

            if summary["useful_table"]:
                table_store.append({
                    "raw": block["content"],
                    "type": "table",
                    "summary": summary["description"],
                    "page": page,
                    "section": block.get("section"),
                })

                print(f"  [table] p{page} → summarized & stored")

            else:
                print(f"  [table] p{page} → ignored (not useful)")

        elif btype == "image":
            context = get_context_window(blocks, i)
            summary = summarize_image(block, context)
            
            if summary.get("useful_image"):
                image_store.append({
                    "raw": block["content"],              # alt-text / caption
                    "type": "image",
                    "summary": summary["description"],   # LLM description
                    "image_path": block.get("image_path"),
                    "page": page,
                    "section": block.get("section"),
                })

                print(f"  [image] p{page} → summarized & stored")

            else:
                print(f"  [image] p{page} → ignored (not useful)")

    full_text = '. '.join([item['content'] for item in text_store])
    full_text = clean_text(full_text)

    final_data = {
        'texts': [full_text],
        'images': [item['summary'] for item in image_store],
        'tables': [item['summary'] for item in table_store]
    }

    return final_data

def get_processed_data(pdf_path):
    blocks = preprocess_data(pdf_path)
    final_data = process_blocks(blocks)
    return final_data

if __name__ == "__main__":
    final_data = get_processed_data(PDF_PATH)
    print(final_data)
