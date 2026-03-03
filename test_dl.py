from docling.document_converter import DocumentConverter

source = "lpebun.pdf"  # file path or URL
converter = DocumentConverter()
doc = converter.convert(source).document

print(doc.export_to_markdown())