import os
import json
from pathlib import Path
from openpecha.config import PECHAS_PATH
from openpecha.utils import download_pecha
from openpecha.core.pecha import  OpenPechaFS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def extract_opfs() -> dict[str, str]:
    """
    Extract opfs from clean_opfs.txt and save to opfs.json
    """
    resource_dir = Path(__file__).parent.parent / "resources"
    clean_opfs_path = resource_dir / "clean_opfs.txt"
    clean_opfs: list[str] = clean_opfs_path.read_text(encoding="utf-8").splitlines()

    # Remove leading and trailing whitespace
    clean_opfs: list[str] = [line.strip() for line in clean_opfs]
    clean_opfs: list[str] = [line for line in clean_opfs if line]

    # There should be 266 opfs
    assert len(clean_opfs) == 266

    # Extract opf id and title
    opfs: dict[str, str] = {}
    for opf in clean_opfs:
        opf_id, opf_title = opf.split(",")[0], opf.split(",")[1]
        opfs[opf_id] = opf_title

    # Save opfs to JSON
    opfs_path = resource_dir / "clean_opfs.json"
    opfs_path.write_text(json.dumps(opfs, ensure_ascii=False, indent=2), encoding="utf-8")


def get_pecha_text(pecha_id: str) -> str:
    """
    Get pecha text from OpenPecha-Data
    """
    pecha_download_path = PECHAS_PATH
    pecha_path = download_pecha(pecha_id, pecha_download_path)

    pecha = OpenPechaFS(pecha_path / f"{pecha_id}.opf", pecha_id)
    pecha_text = pecha.bases
    return pecha_text

if __name__ == "__main__":
    pecha_id = "P000270"
    pecha_text = get_pecha_text(pecha_id)
    print(pecha_text) 
    