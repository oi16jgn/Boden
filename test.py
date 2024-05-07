import preprocessing
import pandas as pd


def extract_file_format(cv):
    if pd.notna(cv) and '.' in cv:
        return cv.split('.')[-1].lower()
    return pd.NA


df = preprocessing.load_applicants()

df['File Format'] = df['CV'].apply(extract_file_format)

format_counts = df['File Format'].value_counts()
print(format_counts)

preprocessing.clear_directory('CV')
