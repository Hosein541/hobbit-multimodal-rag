from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

PIPELINES_DIR = ROOT_DIR / "pipelines"

CHROMA_DIR = ROOT_DIR / "vector_db"

BOOK_DIR = ROOT_DIR / "data"

IMAGE_DIR = BOOK_DIR / "images"

METADATA_DIR = BOOK_DIR / "metadata"

CHAINS_DIR = ROOT_DIR / "chains"