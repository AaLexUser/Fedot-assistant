import os
from pathlib import Path

from ..constants import TEXT_EXTENSIONS


def is_text_file(filename: str | Path):
    _, ext = os.path.splitext(filename)
    return ext.lower() in TEXT_EXTENSIONS
