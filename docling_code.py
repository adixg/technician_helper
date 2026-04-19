# Usage
# python docling_code.py "data/manuals/Century NEMA 42-140 Frame Motor Instruction Leaflet.pdf"
# python docling_code.py input.pdf --output_dir converted_docs

import argparse
from pathlib import Path

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def convert_pdf(source: Path, output_dir: Path, progress_callback=None):

    def update(message: str, pct: int):
        if progress_callback is not None:
            progress_callback("docling", message, pct)

    update("Preparing output folders...", 5)

    output_dir.mkdir(parents=True, exist_ok=True)

    doc_stem = source.stem
    image_dir = output_dir / f"{doc_stem}_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    update("Configuring Docling pipeline...", 10)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    update("Converting PDF to Docling document...", 18)

    result = converter.convert(str(source))
    doc = result.document

    update("Scanning extracted elements...", 24)

    items = list(doc.iterate_items())
    total_items = len(items) if items else 1

    picture_idx = 0
    table_idx = 0
    saved_image_paths = []

    for idx, (element, _level) in enumerate(items, start=1):

        pct = 24 + int((idx / total_items) * 36)   # maps roughly 24 -> 60

        if isinstance(element, PictureItem):
            picture_idx += 1
            filename = f"{doc_stem}-picture-{picture_idx}.png"
            out_path = image_dir / filename

            img = element.get_image(doc)
            if img is not None:
                img.save(out_path, "PNG")
                saved_image_paths.append(
                    f"{doc_stem}_images/{filename}"
                )

            update(
                f"Saved picture {picture_idx}...",
                pct
            )

        elif isinstance(element, TableItem):
            table_idx += 1
            filename = f"{doc_stem}-table-{table_idx}.png"
            out_path = image_dir / filename

            img = element.get_image(doc)
            if img is not None:
                img.save(out_path, "PNG")
                saved_image_paths.append(
                    f"{doc_stem}_images/{filename}"
                )

            update(
                f"Saved table {table_idx}...",
                pct
            )

    update("Exporting markdown...", 70)

    md_path = output_dir / f"{doc_stem}-with-image-refs.md"

    md_text = doc.export_to_markdown(
        image_mode=ImageRefMode.PLACEHOLDER
    )

    update("Injecting image references into markdown...", 82)

    for rel_path in saved_image_paths:
        md_text = md_text.replace(
            "<!-- image -->",
            f"![]({rel_path})",
            1
        )

    md_path.write_text(md_text, encoding="utf-8")

    update("Markdown conversion complete.", 100)

    print(f"Markdown saved to: {md_path.resolve()}")
    print(f"Images saved to:   {image_dir.resolve()}")

    return md_path


def main():

    parser = argparse.ArgumentParser(
        description="Convert PDF to markdown with extracted images."
    )

    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to input PDF file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/manuals_converted",
        help="Directory to save outputs"
    )

    args = parser.parse_args()

    source = Path(args.pdf_path)

    if not source.exists():
        raise FileNotFoundError(f"PDF not found: {source}")

    output_dir = Path(args.output_dir)

    convert_pdf(source, output_dir)


if __name__ == "__main__":
    main()