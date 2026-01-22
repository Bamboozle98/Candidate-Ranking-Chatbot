import json
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity

def save_star(star_path: str, role_keywords: str, candidate_id: int):
    data = {"role_keywords": role_keywords, "starred_id": int(candidate_id)}
    with open(star_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_star(star_path: str):
    with open(star_path, "r", encoding="utf-8") as f:
        return json.load(f)

def rerank_with_star(df_scored: pd.DataFrame, starred_id: int, keywords: str,
                     model_path: str | None = None, alpha: float = 0.4) -> pd.DataFrame:
    """
    Combine base_fit with similarity-to-star in feature space.
    similarity computed using the same preprocessing as the model pipeline if available,
    otherwise using simple TF-IDF over titles.
    """
    out = df_scored.copy()

    # Fallback: similarity in title tf-idf space
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(out["job_title_clean"].fillna(""))

    if starred_id not in set(out["id"]):
        raise ValueError("starred_id not found in current dataframe.")

    star_idx = out.index[out["id"] == starred_id][0]
    star_vec = X[out.index.get_loc(star_idx)]
    sims = cosine_similarity(X, star_vec).ravel()

    out["star_sim"] = sims

    # final score: blend base fit + similarity to star
    out["final_score"] = np.clip(alpha * out["base_fit"] + (1 - alpha) * out["star_sim"], 0, 1)
    return out.sort_values("final_score", ascending=False)
