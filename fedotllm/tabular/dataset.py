import pandas as pd
from pathlib import Path
from typing import TypeAlias
from .constants import (
    CSV_SUFFIXES,
    PARQUET_SUFFIXES,
    EXCEL_SUFFIXES
)
def load_pd(data):
    if isinstance(data, (Path, str)):
        path = data
        if isinstance(path, str):
            path = Path(path)
        format = None
        if path.suffix in EXCEL_SUFFIXES:
            format = "excel"
        elif path.suffix in PARQUET_SUFFIXES:
            format = "parquet"
        elif path.suffix in CSV_SUFFIXES:
            format = "csv"
        else:
            raise Exception("file format " + format + " not supported!")
        
        match format:
            case "excel":
                return pd.read_excel(path, engine="calamine")
            case "parquet":
                try:
                    return pd.read_parquet(path, engine="fastparquet")
                except Exception:
                    return pd.read_parquet(path, engine="pyarrow")
            case "csv":
                return pd.read_csv(path)
    else:
        return pd.DataFrame(data)
    

TabularDataset: TypeAlias = pd.DataFrame
    