import os
import pandas as pd
import preprocessing
from sentence_transformers import SentenceTransformer


def main():
    applicants, applicants_kw = preprocessing.prepare_applicant_data()
    companies = preprocessing.prepare_company_data()

    save_as_n_grams(applicants, applicants_kw, companies)
    save_as_embeddings(applicants, applicants_kw, companies)


def save_as_embeddings(applicants, applicants_kw, companies):
    model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')

    print('Encoding applicants:')
    applicant_embeddings = model.encode(applicants['CV Text'].tolist(), show_progress_bar=True)
    print('Encoding applicants using keywords:')
    applicant_kw = model.encode(applicants_kw['CV Nyckelord'].tolist(), show_progress_bar=True)
    print('Encoding companies:')
    company_embeddings = model.encode(companies['Text'].tolist(), show_progress_bar=True)

    applicant_embeddings_df = pd.DataFrame({
        'id': applicants['Submission ID'],
        'embeddings': list(applicant_embeddings)
    })
    applicant_kw_df = pd.DataFrame({
        'id': applicants['Submission ID'],
        'embeddings': list(applicant_kw)
    })
    company_embeddings_df = pd.DataFrame({
        'id': companies['Submission ID'].astype(str) + ' ' + companies['Namn på företaget'],
        'embeddings': list(company_embeddings)
    })

    applicant_embeddings_path = os.path.join('..', 'text representation', 'applicant_sentence_embeddings.pkl')
    applicant_embeddings_df.to_pickle(applicant_embeddings_path)
    print('Applicant embeddings saved to: applicant_sentence_embeddings.pkl')

    applicant_kw_path = os.path.join('..', 'text representation', 'applicant_kw_embeddings.pkl')
    applicant_kw_df.to_pickle(applicant_kw_path)
    print('Applicant embeddings using keywords saved to: applicant_kw_embeddings.pkl')

    company_embeddings_path = os.path.join('..', 'text representation', 'company_sentence_embeddings.pkl')
    company_embeddings_df.to_pickle(company_embeddings_path)
    print('Company embeddings saved to: company_sentence_embeddings.pkl')


def save_as_n_grams(applicants, applicants_kw, companies):
    print('Generating n-grams for applicants:')
    applicant_ngrams = text_to_ngram_vector(applicants['CV Text'].tolist(), n=3)
    print('Generating n-grams for applicants using keywords:')
    applicant_kw_ngrams = text_to_ngram_vector(applicants_kw['CV Nyckelord'].tolist(), n=3)
    print('Generating n-grams for companies:')
    company_ngrams = text_to_ngram_vector(companies['Text'].tolist(), n=3)

    applicant_ngrams_df = pd.DataFrame({
        'id': applicants['Submission ID'],
        'ngrams': applicant_ngrams
    })
    applicant_kw_ngrams_df = pd.DataFrame({
        'id': applicants_kw['Submission ID'],
        'ngrams': applicant_kw_ngrams
    })
    company_ngrams_df = pd.DataFrame({
        'id': companies['Submission ID'].astype(str) + ' ' + companies['Namn på företaget'],
        'ngrams': company_ngrams
    })

    applicant_ngram_path = os.path.join('..', 'text representation', 'applicant_ngrams.pkl')
    applicant_ngrams_df.to_pickle(applicant_ngram_path)
    print('Applicant n-grams saved to: applicant_ngrams.pkl')

    applicant_kw_ngram_path = os.path.join('..', 'text representation', 'applicant_kw_ngrams.pkl')
    applicant_kw_ngrams_df.to_pickle(applicant_kw_ngram_path)
    print('Applicant n-grams using keywords saved to: applicant_kw_ngrams.pkl')

    company_ngram_path = os.path.join('..', 'text representation', 'company_ngrams.pkl')
    company_ngrams_df.to_pickle(company_ngram_path)
    print('Company n-grams saved to: company_ngrams.pkl')


def generate_n_grams(text, n=3):
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def text_to_ngram_vector(texts, n=3):
    return [set(generate_n_grams(text, n)) for text in texts]


if __name__ == "__main__":
    main()
