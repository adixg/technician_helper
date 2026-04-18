import json
import re
from pathlib import Path

SECTIONS_JSON_PATH = Path(r"data/manuals_sections/Century NEMA 42-140 Frame Motor Instruction Leaflet-sections.json")

# Chunking controls
MAX_CHARS = 1000000
MIN_CHARS = 1


def split_into_paragraphs(text: str):
    text = text.strip()
    if not text:
        return []

    # Split on blank lines first
    paras = re.split(r"\n\s*\n", text)
    paras = [p.strip() for p in paras if p.strip()]
    return paras


def split_long_paragraph(paragraph: str, max_chars: int):
    """
    If a paragraph is too long, split it by sentence boundaries.
    Falls back to hard splitting if needed.
    """
    paragraph = paragraph.strip()
    if len(paragraph) <= max_chars:
        return [paragraph]

    # Sentence-ish split
    sentence_parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9(\-"\'])', paragraph)
    sentence_parts = [s.strip() for s in sentence_parts if s.strip()]

    if len(sentence_parts) <= 1:
        # Fallback: hard split
        chunks = []
        start = 0
        while start < len(paragraph):
            end = min(start + max_chars, len(paragraph))
            chunks.append(paragraph[start:end].strip())
            start = end
        return chunks

    chunks = []
    current = ""

    for sent in sentence_parts:
        if not current:
            current = sent
        elif len(current) + 1 + len(sent) <= max_chars:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_section_text(text: str, max_chars: int, min_chars: int):
    """
    Chunk by paragraphs first, then by sentences if a paragraph is too long.
    """
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return []

    processed_paragraphs = []
    for para in paragraphs:
        processed_paragraphs.extend(split_long_paragraph(para, max_chars))

    chunks = []
    current = ""

    for para in processed_paragraphs:
        if not current:
            current = para
        elif len(current) + 2 + len(para) <= max_chars:
            current += "\n\n" + para
        else:
            chunks.append(current.strip())
            current = para

    if current.strip():
        chunks.append(current.strip())

    # Merge tiny trailing chunks when possible
    merged = []
    i = 0
    while i < len(chunks):
        if i < len(chunks) - 1 and len(chunks[i]) < min_chars:
            combined = chunks[i] + "\n\n" + chunks[i + 1]
            if len(combined) <= max_chars * 1.35:
                merged.append(combined.strip())
                i += 2
                continue
        merged.append(chunks[i])
        i += 1

    return merged


def build_chunks(section_doc: dict):
    out = {
        "source_md_file": section_doc.get("source_md_file"),
        "source_pdf_file": section_doc.get("source_pdf_file"),
        "machine": section_doc.get("machine"),
        "manufacturer": section_doc.get("manufacturer"),
        "manual_type": section_doc.get("manual_type"),
        "num_sections": section_doc.get("num_sections"),
        "chunks": []
    }

    chunk_counter = 0

    for section in section_doc.get("sections", []):
        section_id = section.get("section_id")
        section_title = section.get("section_title", "Untitled")
        section_text = section.get("text", "").strip()
        section_images = section.get("images", [])

        if not section_text and not section_images:
            continue

        text_chunks = chunk_section_text(
            text=section_text,
            max_chars=MAX_CHARS,
            min_chars=MIN_CHARS
        )

        # If section has no text but does have images, keep one empty-ish chunk
        if not text_chunks and section_images:
            text_chunks = [""]

        for idx, chunk_text in enumerate(text_chunks, start=1):
            chunk_counter += 1
            out["chunks"].append({
                "chunk_id": f"chunk_{chunk_counter:04d}",
                "section_id": section_id,
                "section_title": section_title,
                "chunk_index_within_section": idx,
                "chunk_text": chunk_text,
                "images": section_images.copy()
            })

    out["num_chunks"] = len(out["chunks"])
    return out


def main():
    with open(SECTIONS_JSON_PATH, "r", encoding="utf-8") as f:
        section_doc = json.load(f)

    chunk_doc = build_chunks(section_doc)

    # Define output directory
    OUTPUT_DIR = Path("./data/manuals_chunks")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build output filename
    out_filename = SECTIONS_JSON_PATH.stem.replace("-sections", "-chunks") + ".json"
    out_path = OUTPUT_DIR / out_filename

    # Save JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunk_doc, f, indent=2, ensure_ascii=False)

    print(f"Saved chunks JSON to: {out_path}")
    print(f"Total chunks: {chunk_doc['num_chunks']}")


if __name__ == "__main__":
    main()