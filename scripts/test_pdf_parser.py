from pathlib import Path
import fitz
from cognitive_kitchen.ingestion.pdf_parser import PDFRecipeParser


def test_pdf_extraction():
    test_pdf_path = Path("data/raw_pdfs/test_sample.pdf")
    test_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Classic Masala Dosa")
    page.insert_text((50, 100), "Ingredients: 1 cup rice, 1/4 cup urad dal")
    page.insert_text((50, 150), "Step 1: Soak rice and dal for 4 hours.")
    doc.save(str(test_pdf_path))
    doc.close()

    parser = PDFRecipeParser(str(test_pdf_path))
    blocks = parser.extract_blocks()
    full_text = parser.get_full_text()

    assert len(blocks) > 0
    assert blocks[0]["page"] == 1
    assert "Classic Masala Dosa" in full_text
    assert "Ingredients:" in full_text
    assert "Step 1:" in full_text

    if test_pdf_path.exists():
        test_pdf_path.unlink()

    print("All assertions passed.")
    print(f"Extracted block count: {len(blocks)}")
    print(f"Sample block ID: {blocks[0]['block_id']}")


if __name__ == "__main__":
    test_pdf_extraction()