# Usage:
# python .\sections_json_gen.py ".\data\manuals_converted\Standard Induction Motors operation Manual-with-image-refs.md"

import json
import re
import argparse
from pathlib import Path


def parse_md_sections(md_text: str):
    """
    Splits markdown into sections using lines like:
    ## Section Title

    Also collects markdown image refs that appear inside each section.
    """

    lines = md_text.splitlines()

    image_pattern = re.compile(r'!\[\]\((.*?)\)')
    heading_pattern = re.compile(r'^\s*##\s+(.*\S)\s*$')

    sections = []

    current_title = "Preamble"
    current_lines = []
    current_images = []
    section_index = 0

    def flush_section():
        nonlocal section_index, current_title, current_lines, current_images

        text = "\n".join(current_lines).strip()

        if not text and not current_images:
            return

        section_index += 1
        sections.append({
            "section_id": f"section_{section_index:03d}",
            "section_title": current_title,
            "text": text,
            "images": current_images.copy()
        })

    for line in lines:
        heading_match = heading_pattern.match(line)

        if heading_match:
            flush_section()
            current_title = heading_match.group(1).strip()
            current_lines = []
            current_images = []
            continue

        image_matches = image_pattern.findall(line)
        if image_matches:
            current_images.extend(image_matches)

        current_lines.append(line)

    flush_section()
    return sections


def infer_metadata_from_content(md_text: str, md_path: Path):
    return {
        "machine": None,
        "manufacturer": None,
        "manual_type": None
    }


def build_output(
    md_path: Path,
    machine: str | None = None,
    manufacturer: str | None = None,
    manual_type: str | None = None,
):
    md_text = md_path.read_text(encoding="utf-8")

    sections = parse_md_sections(md_text)
    inferred = infer_metadata_from_content(md_text, md_path)

    output = {
        "source_md_file": md_path.name,
        "source_pdf_file": md_path.name.replace("-with-image-refs.md", ".pdf"),
        "machine": machine if machine is not None else inferred["machine"],
        "manufacturer": manufacturer if manufacturer is not None else inferred["manufacturer"],
        "manual_type": manual_type if manual_type is not None else inferred["manual_type"],
        "num_sections": len(sections),
        "sections": sections
    }

    return output


def generate_sections_json(
    md_path: Path,
    output_dir: Path,
    machine: str | None = None,
    manufacturer: str | None = None,
    manual_type: str | None = None,
    progress_callback=None,
):
    def update(message: str, pct: int):
        if progress_callback is not None:
            progress_callback("sections", message, pct)

    update("Reading markdown file...", 35)

    output = build_output(
        md_path=md_path,
        machine=machine,
        manufacturer=manufacturer,
        manual_type=manual_type,
    )

    update("Creating output directory...", 42)

    output_dir.mkdir(parents=True, exist_ok=True)

    out_filename = md_path.stem.replace("-with-image-refs", "-sections") + ".json"
    out_path = output_dir / out_filename

    update("Writing sections JSON...", 48)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    update(f"Sections JSON created with {output['num_sections']} sections.", 50)

    print(f"Saved section JSON to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a markdown manual with image refs into sectioned JSON."
    )

    parser.add_argument(
        "md_path",
        type=str,
        help="Path to the markdown file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/manuals_sections",
        help="Directory to save the JSON output"
    )

    parser.add_argument(
        "--machine",
        type=str,
        default=None,
        help="Optional machine name metadata"
    )

    parser.add_argument(
        "--manufacturer",
        type=str,
        default=None,
        help="Optional manufacturer metadata"
    )

    parser.add_argument(
        "--manual_type",
        type=str,
        default=None,
        help="Optional manual type metadata"
    )

    args = parser.parse_args()

    md_path = Path(args.md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    output_dir = Path(args.output_dir)

    generate_sections_json(
        md_path=md_path,
        output_dir=output_dir,
        machine=args.machine,
        manufacturer=args.manufacturer,
        manual_type=args.manual_type,
    )


if __name__ == "__main__":
    main()