import os
import nltk
import pandas as pd
import requests
import fitz
from keybert import KeyBERT
from nltk.corpus import stopwords
from tqdm import tqdm


def download_pdf(url, directory="CV"):
    response = requests.get(url)
    if response.status_code == 200:
        filename = url.split('/')[-1]
        save_path = os.path.join(directory, filename)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return save_path
    return None


def extract_text_from_pdf(filename):
    try:
        doc = fitz.open(filename)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None


def download_and_extract_text_from_cv(row):
    url = row['CV']
    directory = "CV"
    if not os.path.exists(directory):
        os.makedirs(directory)

    pdf_filename = download_pdf(url, directory)
    if pdf_filename:
        return extract_text_from_pdf(pdf_filename)
    return None


def combine_company_text(row):
    field = row['Branschnamn'] if pd.notna(row['Branschnamn']) else ''
    subfield = row['Specifikt yrkesområde'] if pd.notna(row['Specifikt yrkesområde']) else ''

    if row['Branschnamn'] == 'Annat (fritext)':
        return f"{subfield}."
    else:
        return f"{field}. {subfield}."


def combine_company_text_using_keywords(row):
    keywords = row['Nyckelord'] if pd.notna(row['Nyckelord']) else ''
    field = row['Branschnamn'] if pd.notna(row['Branschnamn']) else ''
    subfield = row['Specifikt yrkesområde'] if pd.notna(row['Specifikt yrkesområde']) else ''

    if row['Branschnamn'] == 'Annat (fritext)':
        return f"{subfield}."
    else:
        return f"{field}. {subfield}."


def combine_applicant_text(row):
    return f"{row['Yrkesområde']}. {row['Specifikt yrkesområde']}."


def load_companies():
    file_path = os.path.join('..', 'data', 'arbetsgivare.csv')
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df.set_index('Submission ID', inplace=True, drop=False)
    return df


def preprocess_company_data(df):
    # Combine and remove sparsely populated columns into a single column 'Specifikt yrkesområde'
    columns_to_combine = [
        'Administration, ekonomi, juridik',
        'Bygg och anläggning', 'Chefer och verksamhetsledare', 'Data/IT',
        'Försäljning, inköp, marknadsföring', 'Hantverksyrken',
        'Hotell, restaurang, storhushåll', 'Hälso- och sjukvård',
        'Industriell tillverkning', 'Installation, drift, underhåll',
        'Kropps- och skönhetsvård', 'Kultur, media, design', 'Militärt arbete',
        'Naturbruk', 'Naturvetenskapligt arbete', 'Pedagogiskt arbete',
        'Sanering och renhållning', 'Socialt arbete', 'Säkerhetsarbete',
        'Tekniskt arbete', 'Transport', 'Branschnamn.1',
        'Jag har ofta brist på personal inom dessa yrkesområden:'
    ]
    df['Specifikt yrkesområde'] = df[columns_to_combine].apply(lambda x: '. '.join(x.dropna()), axis=1)

    return df


def prepare_company_data():
    df = load_companies()
    df = preprocess_company_data(df)

    relevant_columns = ['Branschnamn', 'Specifikt yrkesområde']
    df['Text'] = df.apply(combine_company_text, axis=1)
    df.drop(columns='Specifikt yrkesområde', inplace=True)

    return df


def extract_keywords(description, model, stop_words):
    return model.extract_keywords(description, keyphrase_ngram_range=(1, 3), stop_words=stop_words, top_n=10,
                                  use_mmr=True, diversity=0.5)


def load_applicants():
    file_path = os.path.join('..', 'data', 'arbete-cv.csv')
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df.set_index('Submission ID', inplace=True, drop=False)
    return df


def prepare_applicant_data():
    df = load_applicants()
    df = df.dropna(subset=['CV'])

    tqdm.pandas(desc="Downloading and Extracting CV")
    df['CV Text'] = df.progress_apply(
        lambda row: download_and_extract_text_from_cv(row) if pd.notna(row['CV']) and row['CV'].endswith(
            '.pdf') else pd.NA, axis=1)

    df['CV Text'] = df['CV Text'].str.replace('\d+', '', regex=True)
    df.dropna(subset=['CV Text'], inplace=True)

    df_kw = df.copy()

    kw_model = KeyBERT('paraphrase-multilingual-mpnet-base-v2')

    nltk.download('stopwords')
    swedish_stopwords = stopwords.words('swedish')

    tqdm.pandas(desc="Extracting Keywords")
    df_kw['CV Nyckelord'] = df_kw['CV Text'].progress_apply(
        lambda x: '. '.join([kw[0] for kw in extract_keywords(x, kw_model, swedish_stopwords)]) if pd.notna(x) else x
    )
    df_kw['CV Nyckelord'].dropna(inplace=True)

    for kw in df_kw['CV Nyckelord']:
        print(kw)

    common_ids = df['Submission ID'][df['Submission ID'].isin(df_kw['Submission ID'])]
    df_filtered = df[df['Submission ID'].isin(common_ids)]
    df_kw_filtered = df_kw[df_kw['Submission ID'].isin(common_ids)]

    clear_directory('CV')

    return df_filtered, df_kw_filtered


def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
