from pathlib import Path

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

source = "data/manuals/Century NEMA 42-140 Frame Motor Instruction Leaflet.pdf"
output_dir = Path("data/manuals_converted")
output_dir.mkdir(parents=True, exist_ok=True)

doc_stem = Path(source).stem
image_dir = output_dir / f"{doc_stem}_images"
image_dir.mkdir(parents=True, exist_ok=True)

pipeline_options = PdfPipelineOptions()
pipeline_options.generate_page_images = False
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 2.0

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

result = converter.convert(source)
doc = result.document

picture_idx = 0
table_idx = 0
saved_image_paths = []

for element, _level in doc.iterate_items():
    if isinstance(element, PictureItem):
        picture_idx += 1
        filename = f"{doc_stem}-picture-{picture_idx}.png"
        out_path = image_dir / filename
        img = element.get_image(doc)
        if img is not None:
            img.save(out_path, "PNG")
            saved_image_paths.append(f"{doc_stem}_images/{filename}")

    elif isinstance(element, TableItem):
        table_idx += 1
        filename = f"{doc_stem}-table-{table_idx}.png"
        out_path = image_dir / filename
        img = element.get_image(doc)
        if img is not None:
            img.save(out_path, "PNG")
            saved_image_paths.append(f"{doc_stem}_images/{filename}")

# export markdown without auto-exporting Docling's own image folder
md_path = output_dir / f"{doc_stem}-with-image-refs.md"
md_text = doc.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)

# replace each <!-- image --> placeholder with the saved image paths
for rel_path in saved_image_paths:
    md_text = md_text.replace("<!-- image -->", f"![]({rel_path})", 1)

md_path.write_text(md_text, encoding="utf-8")

print(f"Markdown saved to: {md_path.resolve()}")
print(f"Images saved to:   {image_dir.resolve()}")