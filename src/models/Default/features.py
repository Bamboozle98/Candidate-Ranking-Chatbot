import re
import pandas as pd

def parse_connections(x):
    """Convert '500+' -> 500, '123' -> 123, missing -> 0."""
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s.endswith("+"):
        s = s[:-1]
    s = re.sub(r"[^0-9]", "", s)
    return int(s) if s else 0

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["job_title_clean"] = df["job_title"].apply(clean_text)
    df["location_clean"] = df["location"].apply(clean_text)
    df["connection_num"] = df["connection"].apply(parse_connections)

    # simple flags (works well on small data)
    jt = df["job_title_clean"]
    df["has_aspiring"] = jt.str.contains(r"\baspiring\b", regex=True).astype(int)
    df["has_seeking"] = jt.str.contains(r"\bseeking\b", regex=True).astype(int)
    df["has_manager"] = jt.str.contains(r"\bmanager\b", regex=True).astype(int)
    df["has_engineer"] = jt.str.contains(r"\bengineer\b", regex=True).astype(int)

    return df
