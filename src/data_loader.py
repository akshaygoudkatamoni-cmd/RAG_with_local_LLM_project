from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert into LangChain document structure.
    Supported file types include PDF, TXT, DOCX, JSON, and Excel.
    """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    # PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))

    print(f"[DEBUG] found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file} : {e}")

    # Text files
    text_files = list(data_path.glob('**/*.txt'))

    print(f"[DEBUG] found {len(text_files)} text files: {[str(f) for f in text_files]}")
    
    for text_file in text_files:
        print(f"[DEBUG] Loading text file: {text_file}")
        try:
            loader = TextLoader(str(text_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} text docs from {text_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load text file {text_file} : {e}")

    # JSON files
    json_files = list(data_path.glob('**/*.json'))
    print(f"[DEBUG] found {len(json_files)} JSON files: {[str(f) for f in json_files]}")
    
    for json_file in json_files:
        print(f"[DEBUG] Loading JSON file: {json_file}")
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON file {json_file} : {e}")

    # Excel files
    excel_files = list(data_path.glob('**/*.xlsx'))
    print(f"[DEBUG] found {len(excel_files)} Excel files: {[str(f) for f in excel_files]}")

    for excel_file in excel_files:
        print(f"[DEBUG] Loading Excel file: {excel_file}")
        try:
            loader = UnstructuredExcelLoader(str(excel_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {excel_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Excel file {excel_file} : {e}")

    return documents