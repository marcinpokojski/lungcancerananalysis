import pandas as pd

def load_and_clean_data(data) -> pd.DataFrame:
    # csv for training, df for streamlit
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_csv(data)

    # remove patient_id (PII + no predictive value)
    if 'patient_id' in df.columns:
        df = df.drop(columns=['patient_id'])

    # numeric columns
    numeric_cols = ['age', 'pack_years']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # binary variables (explicit mapping)
    binary_cols = [
        'asbestos_exposure',
        'secondhand_smoke_exposure',
        'copd_diagnosis',
        'family_history',
        'lung_cancer'
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .map({'yes': 1, 'no': 0})
                .fillna(0)
                .astype(int)
            )

    # ordinal features with real ordering
    if 'alcohol_consumption' in df.columns:
        df['alcohol_consumption'] = (
            df['alcohol_consumption']
            .astype(str)
            .str.lower()
            .map({'none': 0, 'moderate': 1, 'heavy': 2})
            .fillna(0)
            .astype(int)
        )

    if 'radon_exposure' in df.columns:
        df['radon_exposure'] = (
            df['radon_exposure']
            .astype(str)
            .str.lower()
            .map({'low': 0, 'medium': 1, 'high': 2})
            .fillna(0)
            .astype(int)
        )

    # gender (binary, not ordinal)
    if 'gender' in df.columns:
        df['gender'] = (
            df['gender']
            .astype(str)
            .str.lower()
            .map({'male': 0, 'female': 1})
            .fillna(0)
            .astype(int)
        )

    return df
