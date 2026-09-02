from technician_helper.ingestion.sections_to_chunks import (
    build_chunks,
    chunk_section_text,
    split_into_paragraphs,
    split_long_paragraph,
)


class TestSplitIntoParagraphs:
    def test_splits_on_blank_lines(self):
        assert split_into_paragraphs("a\n\nb\n\nc") == ["a", "b", "c"]

    def test_empty_text(self):
        assert split_into_paragraphs("   ") == []


class TestSplitLongParagraph:
    def test_short_paragraph_untouched(self):
        assert split_long_paragraph("short one", max_chars=100) == ["short one"]

    def test_long_paragraph_is_broken_up(self):
        para = "Sentence one is here. Sentence two is here. Sentence three is here."
        parts = split_long_paragraph(para, max_chars=30)
        assert len(parts) > 1
        assert all(len(p) <= 30 for p in parts)

    def test_single_unsplittable_sentence_is_hard_wrapped(self):
        para = "x" * 250
        parts = split_long_paragraph(para, max_chars=100)
        assert [len(p) for p in parts] == [100, 100, 50]


class TestChunkSectionText:
    def test_respects_max_chars(self):
        text = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(10))
        chunks = chunk_section_text(text, max_chars=80, min_chars=1)
        assert chunks
        assert all(len(c) <= 80 * 1.35 for c in chunks)

    def test_merges_tiny_leading_chunk(self):
        text = "tiny\n\n" + ("normal paragraph text " * 10)
        chunks = chunk_section_text(text, max_chars=150, min_chars=20)
        assert chunks[0].startswith("tiny")
        assert "normal paragraph text" in chunks[0]


class TestBuildChunks:
    def test_numbers_chunks_and_counts(self):
        section_doc = {
            "sections": [
                {
                    "section_id": "section_001",
                    "section_title": "Intro",
                    "text": "Some intro text.",
                    "images": [],
                },
                {
                    "section_id": "section_002",
                    "section_title": "Details",
                    "text": "More detail here.",
                    "images": ["img/a.png"],
                },
            ]
        }
        out = build_chunks(section_doc, max_chars=1000, min_chars=1)
        assert out["num_chunks"] == len(out["chunks"]) == 2
        assert out["chunks"][0]["chunk_id"] == "chunk_0001"
        assert out["chunks"][1]["images"] == ["img/a.png"]
