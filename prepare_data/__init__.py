import json
from pathlib import Path
from openpecha.config import PECHAS_PATH
from openpecha.utils import download_pecha
from openpecha.core.pecha import  OpenPechaFS
from openpecha.core.metadata import PechaMetadata
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

RESOURCE_DIR = Path(__file__).parent.parent / "resources"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

def extract_pecha_ids() -> dict[str, str]:
    """
    Extract pecha ids from clean_pechas.txt and save to pechas.json
    """
    clean_pechas_path = RESOURCE_DIR / "clean_pechas.txt"
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


def get_pecha_data(pecha_id: str) -> dict[str, str]:
    """
    Get pecha metadata and texts from OpenPecha-Data
    """
    pecha_download_path = PECHAS_PATH
    pecha_path = download_pecha(pecha_id, pecha_download_path)

    pecha = OpenPechaFS(pecha_path / f"{pecha_id}.opf", pecha_id)
    
    # Get metadata
    metadata: PechaMetadata = pecha.meta
    metadata: dict = metadata.json(ensure_ascii=False, indent=2, by_alias=True)
    
    # Get volumes
    volume_ids: list[str] = list(pecha.components.keys())

    # Get volume texts
    volume_texts: dict[str, str] = {}
    for volume_id in volume_ids:
        volume_texts[volume_id] = Path(pecha.base_path / f"{volume_id}.txt").read_text(encoding="utf-8")
    
    # Save pecha data
    pecha_data: dict[str, str] = {
        "pecha_id": pecha_id,
        "metadata": metadata,
        "texts": volume_texts,
    }
    
    return pecha_data

def get_and_save_all_pecha_data():
    pecha_ids_path = RESOURCE_DIR / "clean_pecha_ids.json"
    pecha_ids: dict[str, str] = json.loads(pecha_ids_path.read_text(encoding="utf-8"))
    pecha_ids: list[str] = list(pecha_ids.keys())

    for pecha_id in tqdm(pecha_ids, desc="Getting Pecha Data"):
        pecha_data = get_pecha_data(pecha_id)
        pecha_data_path = OUTPUT_DIR / f"{pecha_id}.json"
        pecha_data_path.write_text(json.dumps(pecha_data, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    get_and_save_all_pecha_data()