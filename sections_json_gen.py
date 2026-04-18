import json
import re
from pathlib import Path

MD_PATH = Path(r"./data/manuals_converted/Century NEMA 42-140 Frame Motor Instruction Leaflet-with-image-refs.md")

# Optional metadata
MACHINE = "Century NEMA 42-140 Frame Motor"
MANUFACTURER = "Century"
MANUAL_TYPE = "instruction_leaflet"


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

        # Skip completely empty sections
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


def build_output(md_path: Path):
    md_text = md_path.read_text(encoding="utf-8")

    sections = parse_md_sections(md_text)

    output = {
        "source_md_file": md_path.name,
        "source_pdf_file": md_path.name.replace("-with-image-refs.md", ".pdf"),
        "machine": MACHINE,
        "manufacturer": MANUFACTURER,
        "manual_type": MANUAL_TYPE,
        "num_sections": len(sections),
        "sections": sections
    }

    return output


def main():
    output = build_output(MD_PATH)

    # Define output directory
    OUTPUT_DIR = Path("./data/manuals_sections")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build output filename
    out_filename = MD_PATH.stem.replace("-with-image-refs", "-sections") + ".json"
    out_path = OUTPUT_DIR / out_filename

    # Save JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved section JSON to: {out_path}")


if __name__ == "__main__":
    main()

