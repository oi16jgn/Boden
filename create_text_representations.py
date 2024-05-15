import os
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing
from sentence_transformers import SentenceTransformer


def main():
    applicants, applicants_kw = preprocessing.prepare_applicant_data()
    companies = preprocessing.prepare_company_data()

    save_as_n_grams(applicants, applicants_kw, companies)

    applicants = applicants.drop(columns=['ngrams'])
    applicants_kw = applicants_kw.drop(columns=['ngrams'])
    companies = companies.drop(columns=['ngrams'])

    save_as_embeddings(applicants, applicants_kw, companies)


def save_as_n_grams(applicants, applicants_kw, companies):
    all_texts = applicants['CV Text'].tolist() + applicants_kw['CV Nyckelord'].tolist() + companies['Text'].tolist()
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
    vectorizer.fit(all_texts)

    applicants['ngrams'] = list(vectorizer.transform(applicants['CV Text'].tolist()).toarray())
    applicants_kw['ngrams'] = list(vectorizer.transform(applicants_kw['CV Nyckelord'].tolist()).toarray())
    companies['ngrams'] = list(vectorizer.transform(companies['Text'].tolist()).toarray())

    applicant_ngrams_path = os.path.join('..', 'text representation', 'applicant_ngrams.pkl')
    applicants.to_pickle(applicant_ngrams_path)
    print('Applicant n-grams saved to: ' + applicant_ngrams_path)

    applicant_kw_ngrams_path = os.path.join('..', 'text representation', 'applicant_kw_ngrams.pkl')
    applicants_kw.to_pickle(applicant_kw_ngrams_path)
    print('Applicant keywords n-grams saved to: ' + applicant_kw_ngrams_path)

    company_ngrams_path = os.path.join('..', 'text representation', 'company_ngrams.pkl')
    companies.to_pickle(company_ngrams_path)
    print('Company n-grams saved to: ' + company_ngrams_path)


def save_as_embeddings(applicants, applicants_kw, companies):
    model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')

    print('Encoding applicants:')
    applicant_embeddings = model.encode(applicants['CV Text'].tolist(), show_progress_bar=True)
    print('Encoding applicants using keywords:')
    applicant_kw = model.encode(applicants_kw['CV Nyckelord'].tolist(), show_progress_bar=True)
    print('Encoding companies:')
    company_embeddings = model.encode(companies['Text'].tolist(), show_progress_bar=True)

    applicants['embeddings'] = list(applicant_embeddings)
    applicants_kw['embeddings'] = list(applicant_kw)
    companies['embeddings'] = list(company_embeddings)

    applicant_embeddings_path = os.path.join('..', 'text representation', 'applicant_sentence_embeddings.pkl')
    applicants.to_pickle(applicant_embeddings_path)
    print('Applicant embeddings saved to: ' + applicant_embeddings_path)

    applicant_kw_path = os.path.join('..', 'text representation', 'applicant_kw_embeddings.pkl')
    applicants_kw.to_pickle(applicant_kw_path)
    print('Applicant embeddings using keywords saved to: ' + applicant_kw_path)

    company_embeddings_path = os.path.join('..', 'text representation', 'company_sentence_embeddings.pkl')
    companies.to_pickle(company_embeddings_path)
    print('Company embeddings saved to: ' + company_embeddings_path)


if __name__ == "__main__":
    main()
