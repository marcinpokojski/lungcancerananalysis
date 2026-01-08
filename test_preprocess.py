from preprocess import load_and_clean_data

df = load_and_clean_data("data/lung_cancer_dataset.csv")
print(df.head())
print(df.dtypes)
