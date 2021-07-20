# stdlib
import urllib.request
from pathlib import Path


def download_if_needed(path: Path, url: str) -> None:
    """
    Helper for downloading a file, if it is now already on the disk.
    """
    if path.exists():
        return

    if url.lower().startswith("http"):
        urllib.request.urlretrieve(url, path)  # nosec
        return

    raise ValueError(f"Invalid url provided {url}")
