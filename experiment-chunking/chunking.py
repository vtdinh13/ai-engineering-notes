
import pdfplumber
import re 
from typing import Optional
import hashlib


# pdf_path = "AI_Engineering_Building_Applications_Chip_Huyen.pdf"

def extract_text_from_pdf(start_page:int, end_page:Optional[int] = None, pdf_path:Optional[str] = None) -> dict:
    if pdf_path is None:
        pdf_path = "AI_Engineering_Building_Applications_Chip_Huyen.pdf"
    
    pages = {}
    with pdfplumber.open(str(pdf_path)) as pdf:

        if end_page is None:
            end_page = len(pdf.pages)
        
        for page_num in range(start_page, end_page):
            page = pdf.pages[page_num-1]
            text = (page.extract_text() or "")
            pages[page_num] = {
                "chapter": 4, 
                "text": text
            }
    return pages


def chunk_paragraphs(pages:dict):
    SECTION_BREAK_RE = re.compile(r"([.!?]\s*\n+)")

    sections = []
    current_section_id = 0

    for page_num, payload in pages.items():
        text = payload["text"]
        paragraphs = SECTION_BREAK_RE.split(text)

        buffer = ""
        for paragraph in paragraphs:
            if SECTION_BREAK_RE.fullmatch(paragraph):
                buffer += paragraph.strip("\n")  # keep punctuation/whitespace
                current_section_id += 1
                sections.append(
                    {
                        "chapter": payload["chapter"],
                        "pages": [page_num],
                        "section": current_section_id,
                        "text": buffer.strip(),
                    }
                )
                buffer = ""
            else:
                buffer += paragraph

        if buffer.strip():
            current_section_id += 1
            sections.append(
                {   
                    "chapter": payload["chapter"],
                    "paragraph": current_section_id,
                    "pages": [page_num],
                    "text": buffer.strip(),
                }
            )
    return sections



def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        chunk = seq[i:i+size]
        result.append({'start': i, 'text': chunk})
        if i + size >= n:
            break

    return result

def chunk_documents(docs:list, size:int, step:int) -> list:
    doc_chunks = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('text')
        chunks = sliding_window(doc_content, size, step)
        for chunk in chunks:
            chunk.update(doc_copy)
        doc_chunks.extend(chunks)
    return doc_chunks


def generate_chunk_id(chunk):
    combined = f"{chunk["start"]}-{chunk["text"][:10]}"
    hash_hex = hashlib.md5(combined.encode()).hexdigest()
    chunk_id = hash_hex[:8]
    return chunk_id


def chunk_sliding_window(pages:dict, size:int, step:int) -> list:
    all_text = []
    for page_num in sorted(pages):
        payload = pages[page_num]
        all_text.append(payload["text"])

    combined = [{
        "chapter": 4,         
        "text": "\n".join(all_text),
    }]

    chunks = chunk_documents(combined, size, step)

    return chunks






# from minsearch import VectorSearch
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from tqdm import tqdm

# embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
# v_index = VectorSearch(keyword_fields = [])

# def create_doc_embeddings(chunks:list):
#     embeddings = []

#     for d in tqdm(chunks):
#         v = embedding_model.encode(d['text'])
#         embeddings.append(v)

#     return np.array(embeddings)


# def create_vector_index(chunks:list):
#     emb_array = create_doc_embeddings(chunks)
#     return v_index.fit(emb_array, chunks)



# def text_embedding_search(query:str):
#     query_embedding = embedding_model.encode(query)
#     return v_index.search(query_embedding, num_results=5)



# vector_store = create_vector_index(chunks)


