# Usage:
# python .\chunks_json_gen.py ".\data\manuals_sections\Standard Induction Motors operation Manual-sections.json"

import json
import re
import argparse
from pathlib import Path


DEFAULT_MAX_CHARS = 1000000
DEFAULT_MIN_CHARS = 1


def split_into_paragraphs(text: str):
    text = text.strip()
    if not text:
        return []

    paras = re.split(r"\n\s*\n", text)
    paras = [p.strip() for p in paras if p.strip()]
    return paras


def split_long_paragraph(paragraph: str, max_chars: int):
    paragraph = paragraph.strip()
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentence_parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9(\-"\'])', paragraph)
    sentence_parts = [s.strip() for s in sentence_parts if s.strip()]

    if len(sentence_parts) <= 1:
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


def build_chunks(section_doc: dict, max_chars: int, min_chars: int):
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
            max_chars=max_chars,
            min_chars=min_chars
        )

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


def generate_chunks_json(
    sections_json_path: Path,
    output_dir: Path,
    max_chars: int,
    min_chars: int,
    progress_callback=None,
):
    def update(message: str, pct: int):
        if progress_callback is not None:
            progress_callback("chunks", message, pct)

    update("Reading sections JSON...", 55)

    with open(sections_json_path, "r", encoding="utf-8") as f:
        section_doc = json.load(f)

    update("Building chunks from sections...", 60)

    chunk_doc = build_chunks(
        section_doc=section_doc,
        max_chars=max_chars,
        min_chars=min_chars
    )

    update("Creating output directory...", 65)

    output_dir.mkdir(parents=True, exist_ok=True)

    out_filename = sections_json_path.stem.replace("-sections", "-chunks") + ".json"
    out_path = output_dir / out_filename

    update("Writing chunks JSON...", 67)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunk_doc, f, indent=2, ensure_ascii=False)

    update(f"Chunks JSON created with {chunk_doc['num_chunks']} chunks.", 68)

    print(f"Saved chunks JSON to: {out_path}")
    print(f"Total chunks: {chunk_doc['num_chunks']}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert sectioned manual JSON into chunked JSON."
    )

    parser.add_argument(
        "sections_json_path",
        type=str,
        help="Path to the input sections JSON file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/manuals_chunks",
        help="Directory to save the output chunks JSON"
    )

    parser.add_argument(
        "--max_chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum characters per chunk"
    )

    parser.add_argument(
        "--min_chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Minimum characters for small-chunk merging"
    )

    args = parser.parse_args()

    sections_json_path = Path(args.sections_json_path)
    if not sections_json_path.exists():
        raise FileNotFoundError(f"Sections JSON file not found: {sections_json_path}")

    output_dir = Path(args.output_dir)

    generate_chunks_json(
        sections_json_path=sections_json_path,
        output_dir=output_dir,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
    )


if __name__ == "__main__":
    main()