import json
from pathlib import Path

def extract_opfs() -> dict[str, str]:
    """
    Extract opfs from clean_opfs.txt and save to opfs.json
    """
    resource_dir = Path(__file__).parent.parent / "resources"
    clean_opfs_path = resource_dir / "clean_opfs.txt"
    clean_opfs: list[str] = clean_opfs_path.read_text(encoding="utf-8").splitlines()

    # Remove leading and trailing whitespace
    clean_opfs: list[str] = [line.strip() for line in clean_opfs]

    # Extract opf id and title
    opfs: dict[str, str] = {}
    for opf in clean_opfs:
        opf_id, opf_title = opf.split(",")[0], opf.split(",")[1]
        opfs[opf_id] = opf_title

    # Save opfs to JSON
    opfs_path = resource_dir / "clean_opfs.json"
    opfs_path.write_text(json.dumps(opfs, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    extract_opfs()

    