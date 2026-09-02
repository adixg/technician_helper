from technician_helper.ingestion.markdown_to_sections import parse_md_sections

SAMPLE_MD = """Some preamble text before any heading.

## Installation

Mount the motor on a level surface.

![](Manual_images/manual-picture-1.png)

## Grounding

Connect the ground lead to the frame.
"""


def test_splits_on_h2_headings():
    sections = parse_md_sections(SAMPLE_MD)
    titles = [s["section_title"] for s in sections]
    assert titles == ["Preamble", "Installation", "Grounding"]


def test_section_ids_are_sequential_and_padded():
    sections = parse_md_sections(SAMPLE_MD)
    assert [s["section_id"] for s in sections] == [
        "section_001",
        "section_002",
        "section_003",
    ]


def test_collects_image_refs_per_section():
    sections = parse_md_sections(SAMPLE_MD)
    by_title = {s["section_title"]: s for s in sections}
    assert by_title["Installation"]["images"] == ["Manual_images/manual-picture-1.png"]
    assert by_title["Grounding"]["images"] == []


def test_empty_document_has_no_sections():
    assert parse_md_sections("") == []
