import re
import hashlib

def clean_text(text: str) -> str:
    """Remove extra whitespace and normalize the text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_text(text: str, chunk_size: int, overlap: int) -> list:
    """Split text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def format_confidence(score: float) -> str:
    """Convert float confidence to readable percentage."""
    return f"{round(score * 100, 2)}%"

def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()