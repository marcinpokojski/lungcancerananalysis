from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def generate_plots(df: pd.DataFrame):
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    gender_map = {0: "Male", 1: "Female"}
    alcohol_map = {0: "None", 1: "Moderate", 2: "Heavy"}

        # wystapienie raka z podzialem na wiek
    if {'lung_cancer', 'age'}.issubset(df.columns):
            cancer_df = df[df['lung_cancer'] == 1]

            age_bins = pd.cut(
                cancer_df['age'],
                bins=[18, 30, 40, 50, 60, 70,80,90,100],
                right=False
            )

            cancer_by_age = age_bins.value_counts().sort_index()

            cancer_by_age.plot(kind="bar")
            plt.title("Lung cancer cases by age group")
            plt.xlabel("Age group")
            plt.ylabel("Number of cancer cases")
            plt.tight_layout()

            filename = "lung_cancer_by_age.png"
            plt.savefig(filename, dpi=300)
            plt.close()

            print(f"Lung cancer by age chart saved to: {filename}")

        # wystapienie raka z podzialem na plec
    if {'lung_cancer', 'gender'}.issubset(df.columns):
            cancer_by_gender = (
                df.groupby('gender')['lung_cancer']
                .mean()
            )

            cancer_by_gender.index = cancer_by_gender.index.map(gender_map)

            cancer_by_gender.plot(kind="bar")
            plt.title("Lung cancer occurrence by gender")
            plt.xlabel("Gender")
            plt.ylabel("Cancer rate")
            plt.tight_layout()

            filename = "lung_cancer_by_gender.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"Lung cancer by gender chart saved to: {filename}")

        # wystapienie raka z uzwzglednieniem spozycia alkoholu przez pacjenta
    if {'lung_cancer', 'alcohol_consumption'}.issubset(df.columns):
            cancer_by_alcohol = (
                df.groupby('alcohol_consumption')['lung_cancer']
                .mean()
            )

            cancer_by_alcohol.index = cancer_by_alcohol.index.map(alcohol_map)

            cancer_by_alcohol.plot(kind="bar")
            plt.title("Lung cancer occurrence by alcohol consumption")
            plt.xlabel("Alcohol consumption")
            plt.ylabel("Cancer rate")
            plt.tight_layout()

            filename = "lung_cancer_by_alcohol.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"Lung cancer by alcohol chart saved to: {filename}")

            # wystapienie raka pluc z podzialem na pack years
    if {'lung_cancer', 'pack_years'}.issubset(df.columns):
                pack_year_bins = pd.cut(
                    df['pack_years'],
                    bins=[0,10,20, 30, 40, 50, 60, 70, 80, 90, 100],
                    right=False
                )

                cancer_by_pack_years = (
                    df.groupby(pack_year_bins)['lung_cancer']
                    .mean()
                )

                cancer_by_pack_years.plot(kind="bar")
                plt.title("Lung cancer occurrence by pack years")
                plt.xlabel("Pack years")
                plt.ylabel("Cancer rate")
                plt.tight_layout()

                filename = "lung_cancer_by_pack_years.png"
                plt.savefig(filename, dpi=300)
                plt.close()

                print(f"Lung cancer by pack years chart saved to: {filename}")

     # rozklad wieku
    if 'age' in df.columns:
            plt.hist(df['age'], bins=20)
            plt.title("Age distribution")
            plt.xlabel("Age")
            plt.ylabel("Count")
            plt.tight_layout()

            filename = "age_distribution.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"Age distribution saved to: {filename}")

    # rozklad plci
    if 'gender' in df.columns:
            gender_counts = df['gender'].value_counts().sort_index()
            gender_counts.index = gender_counts.index.map(gender_map)

            gender_counts.plot(kind="bar")
            plt.title("Gender distribution")
            plt.xlabel("Gender")
            plt.ylabel("Count")
            plt.tight_layout()

            filename = "gender_distribution.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"Gender distribution saved to: {filename}")

    # rozklad pack years
    if 'pack_years' in df.columns:
            plt.hist(df['pack_years'], bins=25)
            plt.title("Pack years distribution")
            plt.xlabel("Pack years")
            plt.ylabel("Count")
            plt.tight_layout()

            filename = "pack_years_distribution.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"Pack years distribution saved to: {filename}")

    # rozklad ekspozycji na azbest
    if 'asbestos_exposure' in df.columns:
            asbestos_counts = df['asbestos_exposure'].value_counts().sort_index()
            asbestos_counts.index = asbestos_counts.index.map({0: "No", 1: "Yes"})

            asbestos_counts.plot(kind="bar")
            plt.title("Asbestos exposure distribution")
            plt.xlabel("Asbestos exposure")
            plt.ylabel("Count")
            plt.tight_layout()

            filename = "asbestos_exposure_distribution.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"Asbestos exposure distribution saved to: {filename}")

    # rozklad COPD
    if 'copd_diagnosis' in df.columns:
            copd_counts = df['copd_diagnosis'].value_counts().sort_index()
            copd_counts.index = copd_counts.index.map({0: "No", 1: "Yes"})

            copd_counts.plot(kind="bar")
            plt.title("COPD diagnosis distribution")
            plt.xlabel("COPD diagnosis")
            plt.ylabel("Count")
            plt.tight_layout()

            filename = "copd_distribution.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"COPD distribution saved to: {filename}")




def load_and_clean_data(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    else:
        raise TypeError(type(data))

# usuniecie kolumny patient_id w celu zachowania poufnosci danych
    if 'patient_id' in df.columns:
        df = df.drop(columns=['patient_id'])

#konwersja na dane numeryczne
    numeric_cols = ['age', 'pack_years']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())


    ## walidacja pack_years -> maksymalnie 3x wiek.
    df = df[df["pack_years"] <= 3 * df["age"]]

    #nadanie kolumna binarnych wartosci, yes->1, no->0
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
                df[col].astype(str)
                .str.lower()
                .map({'yes': 1, 'no': 0})
                .fillna(0)
                .astype(int)
            )

#nadanie kolumnom liczbowych wartosci, none -> 0, moderate -> 1, heavy -> 2
    if 'alcohol_consumption' in df.columns:
        df['alcohol_consumption'] = (
            df['alcohol_consumption']
            .astype(str)
            .str.lower()
            .map({'none': 0, 'moderate': 1, 'heavy': 2})
            .fillna(0)
            .astype(int)
        )
    # nadanie kolumnom liczbowych wartosci, low -> 0, medium -> 1, high -> 2
    if 'radon_exposure' in df.columns:
        df['radon_exposure'] = (
            df['radon_exposure']
            .astype(str)
            .str.lower()
            .map({'low': 0, 'medium': 1, 'high': 2})
            .fillna(0)
            .astype(int)
        )

    # nadanie kolumna licznowych wartosc, male -> 0, female -> 1
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
