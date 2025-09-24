import os
from pathlib import Path
from llama_index.readers.docling import DoclingReader

INPUT_PATH = "../data/docs"
OUTPUT_PATH = "../data/processed_docs/md"

# Create output directory if needed
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Initialize reader
reader = DoclingReader(keep_image=False, md_export_kwargs={"page_break_placeholder":"<!-- page break -->"})

# Get all PDF files
pdf_files = Path(INPUT_PATH).glob("*.pdf")

for pdf_file in pdf_files:
    print(f"Processing {pdf_file.name}...")
    
    # Load document
    doc = reader.load_data([str(pdf_file)])
    
    # Extract markdown content
    md_content = ""
    for document in doc:
        md_content += document.text + "\n\n"
    
    # Create output filename
    output_file = Path(OUTPUT_PATH) / f"{pdf_file.stem}.md"
    
    # Save to markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content.strip())
    
    print(f"Saved {output_file}")

print("All files processed!")