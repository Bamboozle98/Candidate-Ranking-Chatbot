import pandas as pd
from features import add_basic_features
from ranker import initial_filter, score_candidates
from feedback import save_star, rerank_with_star

CSV_PATH = r"C:\Users\cbran\PycharmProjects\8XPTuDF1AleElmm6\data\raw\potential-talents - Aspiring human resources - seeking human resources.csv"
KEYWORDS = "aspiring human resources"
STAR_PATH = "artifacts/starred.json"


def main():
    df = pd.read_csv(CSV_PATH)
    df = add_basic_features(df)

    # 1) filter
    df_f = initial_filter(df, KEYWORDS, tau=0.08)
    print(f"After filter: {len(df_f)} candidates")

    # 2) score (no model yet)
    df_s = score_candidates(df_f, KEYWORDS, model_path=None)

    # show top 10
    print("\nTOP 10 (initial):")
    print(df_s.sort_values("base_fit", ascending=False)[["id","job_title","connection","kw_sim","base_fit"]].head(10))

    # 3) star an id (for demo)
    # replace 123 with the one you chose after manual review
    starred_id = int(input("\nEnter starred candidate id: ").strip())
    save_star(STAR_PATH, KEYWORDS, starred_id)

    # 4) rerank
    df_r = rerank_with_star(df_s, starred_id=starred_id, keywords=KEYWORDS, alpha=0.4)
    print("\nTOP 10 (after star rerank):")
    print(df_r[["id","job_title","connection","base_fit","star_sim","final_score"]].head(10))


if __name__ == "__main__":
    main()
