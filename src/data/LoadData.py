import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_time_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- 1. Month → Quarter ------------------------------------
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    if 'month' in df.columns:
        # normalize first 3 letters of month
        df['month_norm'] = (
            df['month']
            .astype(str)
            .str[:3]
            .str.lower()
        )

        df['month_num'] = df['month_norm'].map(month_map)

        df['quarter'] = pd.cut(
            df['month_num'],
            bins=[0, 3, 6, 9, 12],
            labels=['Q1', 'Q2', 'Q3', 'Q4']
        )

    # --- 2. Day → early/mid/late bins ---------------------------
    if 'day' in df.columns:
        df['day_bin'] = pd.cut(
            df['day'],
            bins=[0, 10, 20, 31],
            labels=['early', 'mid', 'late']
        )

    # --- 3. Drop unwanted columns -------------------------------
    drop_cols = []
    if 'day' in df.columns:
        drop_cols.append('day')
    if 'month_num' in df.columns:
        drop_cols.append('month_num')
    if 'month_norm' in df.columns:
        drop_cols.append('month_norm')

    df = df.drop(columns=drop_cols)

    return df


def load_data():
    # Load data
    df = pd.read_csv(
        r'C:\Users\cbran\PycharmProjects\pPr9YTxPUuY9S8uH\data\raw\term-deposit-marketing-2020.csv'
    )

    # Add binned features
    df = add_time_bins(df)

    # Separate target
    X = df.drop('y', axis=1)
    y = df['y'].map({'no': 0, 'yes': 1})

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Columns to standardize
    num_cols_to_scale = ['balance', 'age']

    # Build ColumnTransformer
    encoder = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('scale', StandardScaler(), num_cols_to_scale),
        ],
        remainder='passthrough'  # leave other numeric cols as-is
    )

    # Optional: inspect raw X with headers
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    # print(X.head())  # small preview instead of full dump

    # Train/test split on the *raw* X DataFrame
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit encoder on train, transform both
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    return X_train, X_test, y_train, y_test, encoder

