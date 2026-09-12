import pymupdf
from typing import List, Dict, Any

class PDFRecipeParser:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_blocks(self) -> List[Dict[str, Any]]:
        doc = pymupdf.open(self.file_path)
        blocks_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # get_text("blocks") returns: (x0, y0, x1, y1, text, block_no, block_type)
            blocks = page.get_text("blocks")
            
            for b in blocks:
                # block_type == 0 means text (1 is image)
                if b[6] == 0:
                    text = b[4].strip()
                    if text:
                        blocks_data.append({
                            "page": page_num + 1,
                            "block_id": b[5],
                            "bbox": (b[0], b[1], b[2], b[3]),
                            "text": text
                        })
                        
        doc.close()
        return blocks_data

    def get_full_text(self) -> str:
        doc = pymupdf.open(self.file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text