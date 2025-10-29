import os
import json
from pathlib import Path
from openpecha.config import PECHAS_PATH
from openpecha.utils import download_pecha
from openpecha.core.pecha import  OpenPechaFS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def extract_pecha_ids() -> dict[str, str]:
    """
    Extract pecha ids from clean_pechas.txt and save to pechas.json
    """
    resource_dir = Path(__file__).parent.parent / "resources"
    clean_pechas_path = resource_dir / "clean_pechas.txt"
    clean_pechas: list[str] = clean_pechas_path.read_text(encoding="utf-8").splitlines()

    # Remove leading and trailing whitespace
    clean_pechas: list[str] = [line.strip() for line in clean_pechas]
    clean_pechas: list[str] = [line for line in clean_pechas if line]

    # There should be 266 opfs
    assert len(clean_pechas) == 266

    # Extract opf id and title
    pecha_ids: dict[str, str] = {}
    for pecha in clean_pechas:
        pecha_id, pecha_title = pecha.split(",")[0], pecha.split(",")[1]
        pecha_ids[pecha_id] = pecha_title

    # Save opfs to JSON
    pecha_ids_path = resource_dir / "clean_pecha_ids.json"
    pecha_ids_path.write_text(json.dumps(pecha_ids, ensure_ascii=False, indent=2), encoding="utf-8")


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
    extract_pecha_ids()
    