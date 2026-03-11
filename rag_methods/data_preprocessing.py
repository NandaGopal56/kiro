import os
from dotenv import load_dotenv
load_dotenv()

print(bool(os.getenv("OPENAI_API_KEY")))

import io
import logging
import warnings
import base64
from collections import Counter

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from unstructured.documents.elements import Image, ListItem, NarrativeText, Table, Title
from unstructured.partition.pdf import partition_pdf

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
            "section": None,
        }

        if isinstance(el, (NarrativeText, Title, ListItem)) and el.text:
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

elements = parse_pdf(PDF_PATH)

blocks   = build_blocks(elements)

type_counts = Counter(b["type"] for b in blocks)
print(f"Total blocks: {len(blocks)} | {dict(type_counts)}")

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
            content="You analyze tables extracted from documents and determine if they contain meaningful information."
        ),
        HumanMessage(content=(
            f"""
                Surrounding context:
                {context}

                Table:
                {html}

                Determine whether this table contains useful structured information.
                If useful, summarize it in 2–4 sentences explaining key values and insights.
                If not useful (empty, formatting-only, repeated header, decorative), mark it as not useful.
            """
        )),
    ]

    result = structured_table_llm.invoke(messages)

    return result.model_dump()


def summarize_image(block, context):

    image_path = block.get("image_path")

    if not image_path:
        return {"useful_image": False, "description": "No image path provided."}

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
        },
        {
            "type": "text",
            "text": f"""
                Surrounding context:
                {context}

                Determine if this image contains meaningful information relevant to the document.
            """
        }
    ]

    messages = [
        SystemMessage(
            content="Analyze document images and determine if they contain useful information."
        ),
        HumanMessage(content=content)
    ]

    result = structured_llm.invoke(messages)

    return result.model_dump()

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


final_data = {
    'texts': full_text,
    'images': [item['summary'] for item in image_store],
    'tables': [item['summary'] for item in table_store]
}