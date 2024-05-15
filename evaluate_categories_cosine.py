import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

industries = [
    "Administration, ekonomi, juridik", "Bygg och anläggning", "Chefer och verksamhetsledare",
    "Data/IT", "Försäljning, inköp, marknadsföring", "Hantverksyrken", "Hotell, restaurang, storhushåll",
    "Hälso- och sjukvård", "Industriell tillverkning", "Installation, drift, underhåll", "Kropps- och skönhetsvård",
    "Kultur, media, design", "Militärt arbete", "Naturbruk", "Naturvetenskapligt arbete",
    "Pedagogiskt arbete", "Sanering och renhållning",
    "Socialt arbete", "Säkerhetsarbete", "Tekniskt arbete", "Transport"
]


def main():
    applicants_df = pd.read_pickle('../text representation/applicant_sentence_embeddings.pkl')
    applicants_kw_df = pd.read_pickle('../text representation/applicant_kw_embeddings.pkl')
    companies_df = pd.read_pickle('../text representation/company_sentence_embeddings.pkl')

    applicant_result_df = get_top_matches(applicants_df, companies_df, 'embeddings')
    applicant_kw_result_df = get_top_matches(applicants_kw_df, companies_df, 'embeddings')

    total = total_matches(applicants_df, companies_df)
    print(total)

    result_1 = amount_of_matches(applicants_df, companies_df, applicant_result_df)
    result_2 = amount_of_matches(applicants_df, companies_df, applicant_kw_result_df)
    result_1 = result_1 / total
    result_2 = result_2 / total
    print(f"Sentence embeddings result on whole CV: {result_1:.4f}")
    print(f"Sentence embeddings result on keywords of CV: {result_2:.4f}")

    applicants_df = pd.read_pickle('../text representation/applicant_ngrams.pkl')
    applicants_kw_df = pd.read_pickle('../text representation/applicant_kw_ngrams.pkl')
    companies_df = pd.read_pickle('../text representation/company_ngrams.pkl')

    applicant_result_df = get_top_matches(applicants_df, companies_df, 'ngrams')
    applicant_kw_result_df = get_top_matches(applicants_kw_df, companies_df, 'ngrams')

    result_1 = amount_of_matches(applicants_df, companies_df, applicant_result_df)
    result_2 = amount_of_matches(applicants_df, companies_df, applicant_kw_result_df)
    result_1 = result_1 / total
    result_2 = result_2 / total
    print(f"n-grams result on whole CV: {result_1:.4f}")
    print(f"n-grams result on keywords of CV: {result_2:.4f}")


def check_match(row_applicant, row_company):
    for industry in industries:
        titles_applicant = row_applicant[industry]
        titles_applicant = "" if pd.isna(titles_applicant) else titles_applicant
        if len(titles_applicant) == 0:
            continue
        titles_applicant_list = titles_applicant.split(", ")

        titles_company = row_company[industry]
        titles_company = "" if pd.isna(titles_company) else titles_company
        if len(titles_company) == 0:
            continue
        titles_company_list = titles_company.split(", ")

        for title in titles_applicant_list:
            if title in titles_company_list:
                return True
    return False


def get_top_matches(applicants_df, companies_df, option):
    applicant_embeddings = np.stack(applicants_df[option].values)
    company_embeddings = np.stack(companies_df[option].values)

    similarity_matrix = cosine_similarity(applicant_embeddings, company_embeddings)

    top_20_indices = []
    applicant_ids = []

    for idx, similarities in enumerate(similarity_matrix):
        top_indices = np.argsort(similarities)[-20:][::-1]

        top_20_indices.append(companies_df['Submission ID'].iloc[top_indices].values.tolist())
        applicant_ids.append(applicants_df['Submission ID'].iloc[idx])

    results_df = pd.DataFrame({
        'ID of Applicant': applicant_ids,
        'Top 20 Similar IDs from Companies': top_20_indices
    })

    return results_df


def total_matches(applicants_df, companies_df):
    amount = 0
    for index_applicant, row_applicant in applicants_df.iterrows():
        for index_company, row_company in companies_df.iterrows():
            if check_match(row_applicant, row_company):
                amount += 1

    return amount


def amount_of_matches(applicants_df, companies_df, results_df):
    amount = 0
    for index_applicant, result_row in results_df.iterrows():
        applicant_row = applicants_df.loc[result_row['ID of Applicant']]
        company_ids = result_row['Top 20 Similar IDs from Companies']
        for company_id in company_ids:
            row_company = companies_df.loc[company_id]
            if check_match(applicant_row, row_company):
                amount += 1
            else:
                break

    return amount


if __name__ == "__main__":
    main()
