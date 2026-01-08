import pandas as pd

def load_and_clean_data(data) -> pd.DataFrame:
    #csv for training, df for streamlit
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_csv(data)

    #removing patinet_id for data security
    if 'patient_id' in df.columns:
        df = df.drop(columns=['patient_id'])

    #convert to number, if missing -> 0.
    if 'pack_years' in df.columns:
        df['pack_years'] = pd.to_numeric(df['pack_years'], errors='coerce')
        df['pack_years'] = df['pack_years'].fillna(0).astype(int)

    #binary coding
    binary_mappings = {
        'asbestos_exposure': {'yes': 1, 'no': 0},
        'copd_diagnosis': {'yes': 1, 'no': 0},
        'family_history': {'yes': 1, 'no': 0},
        'secondhand_smoke_exposure': {'yes': 1, 'no': 0},
        'lung_cancer': {'yes': 1, 'no': 0},
    }

    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(mapping)
            df[col] = df[col].fillna(0).astype(int)

    if 'alcohol_consumption' in df.columns:
        df['alcohol_consumption'] = df['alcohol_consumption'].astype(str).str.lower()
        df['alcohol_consumption'] = df['alcohol_consumption'].map({
            'none': 0,
            'moderate': 1,
            'heavy': 2
        }).fillna(0).astype(int)

    if 'radon_exposure' in df.columns:
        df['radon_exposure'] = df['radon_exposure'].astype(str).str.lower()
        df['radon_exposure'] = df['radon_exposure'].map({
            'low': 0,
            'medium': 1,
            'high': 2
        }).fillna(0).astype(int)

    return df
