import re

def clean_text(text: str) -> str:
    # Remove long dash lines
    text = re.sub(r'-{3,}', ' ', text)
    
    # Remove emojis and special symbols (keep normal punctuation)
    text = re.sub(r'[^\w\s\.\,\-\%\(\)]', ' ', text)
    
    # Remove standalone dots
    text = re.sub(r'\n\s*\.\s*\n', '\n', text)
    
    # Fix broken decimals like 10 \n .5-12 \n .5%
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def clean_chunk(text: str) -> str:
    # Remove long dash separators
    text = re.sub(r'-{3,}', ' ', text)

    # Remove "Print to PDF"
    text = re.sub(r'Print to PDF', '', text)

    # Remove leading dot
    text = re.sub(r'^\s*\.\s*', '', text)

    # Fix broken decimal formatting
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)

    # Remove duplicate fragment like "and drying, green coffee beans..."
    text = re.sub(r'and drying, green coffee beans.*?\)', '', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()