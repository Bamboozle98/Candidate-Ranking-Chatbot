import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import plotly.express as px
from src.data.LoadData import add_time_bins
from scipy.stats import chi2_contingency, mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ------------- Data Loading ------------- #
@st.cache_data
def load_data(uploaded_file=None, fallback_path: str = "data/raw/term-deposit-marketing-2020.csv"):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    csv_path = Path(fallback_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    st.error("Please upload the dataset using the sidebar.")
    return None


@st.cache_data
def load_dataset_for_app(uploaded_file=None, fallback_path: str = "data/raw/term-deposit-marketing-2020.csv"):
    # Load the dataset with same preprocessing as the model.
    # This is for the interactive model prediction on the last page.
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        csv_path = Path(fallback_path)
        if not csv_path.exists():
            st.error(
                f"No file uploaded and fallback file '{fallback_path}' not found."
            )
            return None
        df = pd.read_csv(csv_path)

    df = add_time_bins(df)

    return df


def get_column_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    return numeric_cols, categorical_cols


# ------------- Stat Helper Functions ------------- #
def compute_cramers_v(df: pd.DataFrame, categorical_cols, target_col="y"):
    # Compute Cramér's V for each categorical feature vs a categorical target y.
    rows = []
    for col in categorical_cols:
        if col == target_col:
            continue

        ct = pd.crosstab(df[col], df[target_col])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue

        chi2, p, dof, exp = chi2_contingency(ct)
        n = ct.to_numpy().sum()
        if n == 0:
            continue

        phi2 = chi2 / n
        r, k = ct.shape
        denom = min(k - 1, r - 1)
        if denom <= 0:
            continue

        v = np.sqrt(phi2 / denom)

        rows.append(
            {
                "feature": col,
                "cramers_v": v,
                "chi2_p_value": p,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "cramers_v", "chi2_p_value"])

    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False)


def compute_vif(df: pd.DataFrame, numeric_cols):
    # Compute VIF for each numeric feature.
    if not numeric_cols:
        return pd.DataFrame(columns=["feature", "VIF"])

    X = df[numeric_cols].copy().dropna()
    if X.shape[0] < 2:
        return pd.DataFrame(columns=["feature", "VIF"])

    # Add intercept
    X_const = sm.add_constant(X, has_constant="add")

    vifs = []
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        try:
            vif_val = variance_inflation_factor(X_const.values, i)
        except Exception:
            vif_val = np.nan
        vifs.append({"feature": col, "VIF": vif_val})

    return pd.DataFrame(vifs)


def compute_numeric_effects_vs_target(df: pd.DataFrame, numeric_cols, target_col="y"):
    # Compute means, Mann-Whitney U, p-value, and Cliff's Delta for all numeric features.
    if target_col not in df.columns:
        return pd.DataFrame()

    # Get binary groups
    groups = df[target_col].dropna().unique()
    if len(groups) != 2:
        # Only defined for binary targets
        return pd.DataFrame()

    g1, g2 = groups[0], groups[1]

    rows = []
    for col in numeric_cols:
        x1 = df.loc[df[target_col] == g1, col].dropna()
        x2 = df.loc[df[target_col] == g2, col].dropna()

        if len(x1) == 0 or len(x2) == 0:
            continue

        # Mann–Whitney U (U-Stat)
        U, p = mannwhitneyu(x1, x2, alternative="two-sided")

        # Cliff's Delta from U:
        #   delta = 2U / (n1 * n2) - 1
        n1, n2 = len(x1), len(x2)
        delta = (2 * U) / (n1 * n2) - 1

        rows.append(
            {
                "feature": col,
                f"mean({target_col}={g1})": x1.mean(),
                f"mean({target_col}={g2})": x2.mean(),
                "U_stat": U,
                "U_p_value": p,
                "cliffs_delta": delta,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


@st.cache_resource
def load_trained_mlp(
    model_path: str = "src/models/Default/Saved model states/models/mlp_best.joblib",
    encoder_path: str = "src/models/Default/Saved model states/encoders/encoder.joblib",
):
    # Load the already-trained MLP model and encoder.
    model_file = Path(model_path)
    enc_file = Path(encoder_path)

    if not model_file.exists() or not enc_file.exists():
        raise FileNotFoundError(
            f"Model or encoder file not found. "
            f"Expected:\n- {model_file}\n- {enc_file}"
        )

    mlp = joblib.load(model_file)
    encoder = joblib.load(enc_file)
    return mlp, encoder


def pretty(col):
    # Clean up columns pulled from the dataset.
    return col.replace("_", " ").title()


def sci_notation(df, cols, precision=3):
    # Convert numerics to scientific notation for readability.
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: f"{x:.{precision}e}" if pd.notnull(x) else x)
    return df


# ------------- Streamlit ------------- #
def main():
    st.set_page_config(
        page_title="Term Deposit Dataset Explorer",
        layout="wide"
    )

    st.title("📊 Term Deposit Dataset Explorer")

    st.markdown("""
    Explore the **Term Deposit Marketing** dataset using **summary statistics**  
    and **interactive scatter plots** (2D + 3D).  
    **No raw data is displayed in this app.**
    """)

    df = load_data()
    df_2 = load_dataset_for_app()

    if df is None:
        st.stop()

    # Validate target column
    target_col = "y"
    if target_col not in df.columns:
        st.error(f"This dataset must contain a column named `{target_col}` indicating subscription.")
        st.stop()

    numeric_cols, categorical_cols = get_column_types(df)

    # ===================== TABS ===================== #
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Overview"

    tab_choice = st.radio(
        "Navigate:",
        ["Overview", "Summary", "2D Scatter", "3D Scatter", "Model"],
        horizontal=True,
        key="active_tab"
    )

    # ===================== OVERVIEW ===================== #
    if tab_choice == "Overview":
        st.subheader("Dataset Overview")
        st.markdown("This page provides a high-level snapshot of the dataset, highlighting its size, structure, "
                    "and overall composition. The metrics at the top summarize how many data entries and features are available, "
                    "while the accompanying column overview describes the data types, missing values, and unique value counts across all features. "
                    "The pie chart illustrates the distribution of the target variable, subscribers versus non-subscribers, demonstrating a clear imbalance between the two classes. "
                    "The following pages will prvoide a greater understanding of the dataset’s shape before performing deeper analysis.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Numeric Columns", len(numeric_cols))

        st.markdown("### Column Overview Summary")

        overview_rows = []
        for col in df.columns:
            series = df[col]

            row = {
                "column": col,
                "dtype": str(series.dtype),
                "missing_count": series.isna().sum(),
                "unique_values": series.nunique()
            }

            if pd.api.types.is_numeric_dtype(series):
                row.update({
                    "min": series.min(),
                    "max": series.max(),
                    "mean": round(series.mean(), 3),
                    "median": round(series.median(), 3),
                })
            else:
                row.update({
                    "min": None,
                    "max": None,
                    "mean": None,
                    "median": None,
                })

            overview_rows.append(row)

        overview_df = pd.DataFrame(overview_rows).set_index("column")
        overview_df = overview_df.rename(columns={
            "dtype": "Data Type",
            "missing_count": "Missing Values (Count)",
            "unique_values": "Unique Values (How many different values)",
            "min": "Lowest Value",
            "max": "Highest Value",
            "mean": "Average",
            "median": "Middle Value (Median)",
        })

        st.dataframe(overview_df, use_container_width=True)

        st.markdown("### Target Variable Distribution (Subscribers vs Non-Subscribers)")

        counts = df[target_col].value_counts()
        labels = counts.index.tolist()
        values = counts.values.tolist()

        fig_pie = px.pie(
            names=labels,
            values=values,
            title=f"Subscription Breakdown ({target_col})",
            hole=0.35,
        )

        # Make the pie chart bigger
        fig_pie.update_traces(
            textinfo="label+percent",
            textfont_size=18,
            pull=[0.04 if lbl == labels[0] else 0 for lbl in labels]
        )

        fig_pie.update_layout(
            height=600,
            width=600,
            legend_title_text="Subscription Status",
            title_x=0.5,
            margin=dict(t=50, b=50, l=10, r=10)
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    # ===================== SUMMARY STATS ===================== #
    elif tab_choice == "Summary":
        st.subheader("Summary Statistics & Relationships")

        st.markdown(
            "This page summarizes the dataset in plain language. "
            "**Numeric columns** show common summaries (average, median, lowest/highest values, and spread). "
            "Then we compare each feature to the outcome (`y`) to see which ones look meaningfully different between "
            "subscribers and non-subscribers. "
            "For categorical features, we report how strongly each category is associated with subscribing."
        )

        # ---- Overall numeric summary ----
        if numeric_cols:
            st.markdown("#### Overall Numeric Summary (Easy-to-read)")

            numeric_summary = df[numeric_cols].describe().T.rename(columns={
                "count": "Non-Missing Count",
                "mean": "Average",
                "std": "Spread (Std Dev)",
                "min": "Lowest Value",
                "25%": "25th Percentile (Lower Quartile)",
                "50%": "Median (Middle Value)",
                "75%": "75th Percentile (Upper Quartile)",
                "max": "Highest Value",
            })

            numeric_summary.index = numeric_summary.index.map(pretty)
            numeric_summary.index.name = "Numeric Feature"

            st.dataframe(numeric_summary, use_container_width=True)
        else:
            st.info("No numeric columns found for overall summary.")

        # ---- Numeric vs Target: U-Stats, Cliff's Delta, VIF ----
        st.markdown("#### Numeric Features vs Subscription Outcome (Group Differences)")

        num_effects = compute_numeric_effects_vs_target(df, numeric_cols, target_col=target_col)

        # VIF for numeric features (not currently displayed, but computed here if you want it later)
        vif_df = compute_vif(df, numeric_cols)

        if not num_effects.empty:
            # figure out the two target labels used inside compute_numeric_effects_vs_target
            groups = df[target_col].dropna().unique()
            g1, g2 = groups[0], groups[1]

            num_effects_fmt = sci_notation(
                num_effects,
                cols=["U_stat", "U_p_value", "cliffs_delta"]
            ).rename(columns={
                f"mean({target_col}={g1})": f"Average (y = {g1})",
                f"mean({target_col}={g2})": f"Average (y = {g2})",
                "U_stat": "Mann–Whitney Score (Group Difference)",
                "U_p_value": "P-Value (Chance this is random)",
                "cliffs_delta": "Effect Size (How different are groups?)",
            })

            # Make feature names friendly
            num_effects_fmt["feature"] = num_effects_fmt["feature"].apply(pretty)

            st.dataframe(
                num_effects_fmt.set_index("feature").rename_axis("Numeric Feature"),
                use_container_width=True
            )
        else:
            st.info(
                "Could not compute group-difference statistics. "
                "Make sure the target column is binary and numeric features exist."
            )

        # ---- Categorical vs Target: Cramér's V ----
        st.markdown("#### Categorical Features vs Subscription Outcome (Association Strength)")

        if categorical_cols:
            cramers_df = compute_cramers_v(df, categorical_cols, target_col=target_col)

            if not cramers_df.empty:
                cramers_fmt = sci_notation(
                    cramers_df,
                    cols=["chi2_p_value"]
                ).rename(columns={
                    "cramers_v": "Association Strength (0 = none, 1 = strong)",
                    "chi2_p_value": "P-Value (Chance this is random)",
                })

                cramers_fmt["feature"] = cramers_fmt["feature"].apply(pretty)

                st.dataframe(
                    cramers_fmt.set_index("feature").rename_axis("Categorical Feature"),
                    use_container_width=True
                )
            else:
                st.info(
                    "No valid categorical features for association testing "
                    "(need at least 2 category levels and 2 outcome levels)."
                )
        else:
            st.info("No categorical columns found for association testing.")


    # ===================== 2D SCATTER ===================== #
    elif tab_choice == "2D Scatter":
        st.subheader("Interactive 2D Scatter Plots")
        st.markdown("This page allows users to visualize relationships between two numeric features through customizable 2D scatter plots. "
                    "Users can select the X- and Y-axes and optionally color points by the target variable or any other feature. "
                    "This helps reveal potential clusters, linear or nonlinear patterns, and separability between subscribers and non-subscribers. "
                    "By experimenting with different variable combinations, users can explore data trends that may not be immediately obvious from summary statistics alone.")

        if len(numeric_cols) < 2:
            st.warning("You need at least two numeric columns.")
        else:
            # Build display mappings
            x_options = {pretty(c): c for c in numeric_cols}
            y_options = {pretty(c): c for c in numeric_cols}
            c_options = {pretty(c): c for c in [target_col] + numeric_cols + categorical_cols}

            # Selectboxes
            x_label = st.selectbox("X-axis", x_options.keys(), key="2d_x")
            y_label = st.selectbox("Y-axis", y_options.keys(), key="2d_y")
            c_label = st.selectbox("Color by", c_options.keys(), key="2d_c")

            # Recover original column names
            x_axis = x_options[x_label]
            y_axis = y_options[y_label]
            color = c_options[c_label]

            fig2d = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=color,
                title=f"2D Scatter: {x_axis} vs {y_axis}",
                opacity=0.7,
            )

            st.plotly_chart(fig2d, use_container_width=True)

    # ===================== 3D SCATTER ===================== #
    elif tab_choice == "3D Scatter":
        st.subheader("Interactive 3D Scatter Plots")
        st.markdown("The 3D Scatter Plot page expand visual exploration into a three-dimensional space, allowing insight into how multiple features interact simultaneously. "
                    "Users can freely rotate, pan, zoom, and inspect points to examine patterns or separations between subscription classes. "
                    "Controls on the sidebar make it easy to switch axes and color mappings, enabling quick comparisons across many variable combinations. "
                    )

        if len(numeric_cols) < 3:
            st.warning("You need at least three numeric columns.")
        else:
            # Build display → actual column mappings
            x3d_opts = {pretty(c): c for c in numeric_cols}
            y3d_opts = {pretty(c): c for c in numeric_cols}
            z3d_opts = {pretty(c): c for c in numeric_cols}
            c3d_opts = {pretty(c): c for c in [target_col] + numeric_cols + categorical_cols}

            # Selectboxes
            x3d_label = st.selectbox("X-axis", x3d_opts.keys(), key="3d_x")
            y3d_label = st.selectbox("Y-axis", y3d_opts.keys(), key="3d_y")
            z3d_label = st.selectbox("Z-axis", z3d_opts.keys(), key="3d_z")
            c3d_label = st.selectbox("Color by", c3d_opts.keys(), key="3d_c")

            # Recover original column names
            x_axis = x3d_opts[x3d_label]
            y_axis = y3d_opts[y3d_label]
            z_axis = z3d_opts[z3d_label]
            color = c3d_opts[c3d_label]

            fig3d = px.scatter_3d(
                df,
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color=color,
                title=f"3D Scatter: {x_axis} / {y_axis} / {z_axis}",
                opacity=0.7,
            )
            fig3d.update_traces(marker=dict(size=4))

            fig3d.update_layout(
                height=800,
                width=1200,
                margin=dict(t=50, l=0, r=0, b=0)
            )

            st.plotly_chart(fig3d, use_container_width=True)

    # ===================== MODEL PLAYGROUND ===================== #
    elif tab_choice == "Model":
        if "model_widget_version" not in st.session_state:
            st.session_state.model_widget_version = 1

        if st.button("Reset Model Inputs"):
            st.session_state.model_widget_version += 1

        st.subheader("🤖 Model Playground: What-if Prediction")
        st.markdown("The Model Playground enables users to experiment with predictive modeling using an imaginary customer profile. "
                    "All sliders and dropdowns correspond to the same preprocessed features used during model training. "
                    "After the user defines a synthetic input, the saved MLP model processes the data and predicts whether the hypothetical customer would subscribe, alongside class probabilities. "
                    "This page serves as an interactive “what-if” tool, helping us understand how individual variables influence model predictions and allowing them to test different scenarios without exposing any real training data.")

        st.markdown(
        """
        Create an **imaginary customer** and let the pre-trained MLP model predict whether they would subscribe (`y`).
        """
        )

        # Load pre-trained MLP + encoder
        try:
            mlp, encoder = load_trained_mlp()
        except Exception as e:
            st.error(
                "Could not load the trained model / encoder. "
                "Make sure you've run the training script and the files exist.\n\n"
                f"Details: {e}"
            )
            st.stop()

        # Use the preprocessed df schema to match what encoder expects
        target_col = "y"
        feature_cols = [c for c in df_2.columns if c != target_col]

        st.markdown("#### 1. Define a Synthetic Customer")

        user_inputs = {}
        col_left, col_right = st.columns(2)

        ver = st.session_state.model_widget_version

        for i, col in enumerate(feature_cols):
            series = df_2[col]
            container = col_left if i % 2 == 0 else col_right
            display_name = pretty(col)
            key_base = f"model_v{ver}_{col}"

            with container:
                if pd.api.types.is_numeric_dtype(series):
                    col_min = float(series.min())
                    col_max = float(series.max())
                    col_mean = float(series.mean())

                    if pd.api.types.is_integer_dtype(series):
                        user_val = st.slider(
                            label=display_name,
                            min_value=int(col_min),
                            max_value=int(col_max),
                            value=int(round(col_mean)),
                            key=key_base,
                        )
                    else:
                        user_val = st.slider(
                            label=display_name,
                            min_value=col_min,
                            max_value=col_max,
                            value=col_mean,
                            key=key_base,
                        )
                else:
                    unique_vals = sorted(series.dropna().unique().tolist())
                    if not unique_vals:
                        user_val = st.text_input(display_name, "", key=key_base)
                    else:
                        # Map pretty label -> raw value
                        option_map = {pretty(v): v for v in unique_vals}

                        user_label = st.selectbox(
                            label=display_name,
                            options=option_map.keys(),
                            index=0,
                            key=key_base,
                        )
                        # Recover raw value for the model
                        user_val = option_map[user_label]

            user_inputs[col] = user_val

        st.markdown("#### 2. Predict Subscription Outcome")

        if st.button("Predict with MLP"):
            # Build a single-row DataFrame with the SAME columns as training X
            new_row = pd.DataFrame([user_inputs])

            X_new = encoder.transform(new_row)
            if hasattr(X_new, "toarray"):
                X_new = X_new.toarray()

            pred = mlp.predict(X_new)[0]

            prob_text = ""
            if hasattr(mlp, "predict_proba"):
                proba = mlp.predict_proba(X_new)[0]
                class_probs = dict(zip(mlp.classes_, proba))
                prob_text = " | ".join(
                    [f"P(y={cls}) = {p:.3f}" for cls, p in class_probs.items()]
                )

            # --- helper: map model output to True/False + friendly text ---
            def as_bool_and_label(pred):
                if isinstance(pred, str):
                    p = pred.strip().lower()
                    if p in {"yes", "y", "true", "1"}:
                        return True, "Yes"
                    if p in {"no", "n", "false", "0"}:
                        return False, "No"
                if isinstance(pred, (int, float)):
                    return (int(pred) == 1), ("Yes" if int(pred) == 1 else "No")
                if isinstance(pred, bool):
                    return pred, ("Yes" if pred else "No")
                # fallback
                return False, str(pred)

            will_subscribe, yn_label = as_bool_and_label(pred)

            # --- probabilities (optional) ---
            p_yes = None
            p_no = None
            if hasattr(mlp, "predict_proba"):
                proba = mlp.predict_proba(X_new)[0]
                class_probs = dict(zip(mlp.classes_, proba))

                # common class labels
                if "yes" in class_probs:
                    p_yes = float(class_probs["yes"])
                    p_no = float(class_probs.get("no", 1 - p_yes))
                elif 1 in class_probs:
                    p_yes = float(class_probs[1])
                    p_no = float(class_probs.get(0, 1 - p_yes))

            # --- friendly output ---
            st.markdown("---")
            st.markdown("### 🧾 Prediction Result")

            if will_subscribe:
                st.success("**Prediction: This customer would subscribe.**")
            else:
                st.info("**Prediction: This customer would *not* subscribe.**")

            # Nice paragraph explanation
            if p_yes is not None:
                st.write(
                    f"Based on the values you selected, the model thinks the customer **"
                    f"{'would' if will_subscribe else 'would not'} subscribe** "
                    f"(True/False = **{will_subscribe}**). "
                    f"It estimates about **{p_yes:.1%}** chance of subscribing "
                    f"and **{p_no:.1%}** chance of not subscribing."
                )
            else:
                st.write(
                    f"Based on the values you selected, the model thinks the customer **"
                    f"{'would' if will_subscribe else 'would not'} subscribe** "
                    f"(True/False = **{will_subscribe}**)."
                )

            st.caption(
                "This is a model-based estimate from a pre-trained MLP using the same preprocessing pipeline as training."
            )


if __name__ == "__main__":
    main()
