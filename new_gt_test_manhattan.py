import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances

ground_truth = {
    2385: [1249, 1731, 1741],
    1378: [1147, 258],
    967: [839, 1741],
    992: [692, 914],
    1824: [792, 2084, 1746, 1328, 1323, 1005, 894, 754, 747, 707, 704, 691],
    1922: [2941, 1994, 1956, 1314, 1080, 1035],
    1897: [694, 2006, 1960, 258, 1147],
    2771: [258, 1147, 903, 785],
    1820: [2589, 1337, 1327, 1314, 731, 839, 1741],
    1428: [1994, 2692, 2589, 2548, 1337, 948, 696, 1741]
}

def main():
    applicants_df = pd.read_pickle('../text representation/applicant_sentence_embeddings.pkl')
    applicants_kw_df = pd.read_pickle('../text representation/applicant_kw_embeddings.pkl')
    companies_df = pd.read_pickle('../text representation/company_sentence_embeddings.pkl')

    applicant_result_df = get_top_matches(applicants_df, companies_df, 'embeddings')
    applicant_kw_result_df = get_top_matches(applicants_kw_df, companies_df, 'embeddings')

    total = sum(len(companies) for companies in ground_truth.values())
    print("Total number of items:", total)

    result_1 = amount_of_matches(applicant_result_df)
    result_2 = amount_of_matches(applicant_kw_result_df)
    result_1 = result_1 / total
    result_2 = result_2 / total
    print(f"Sentence embeddings result on whole CV: {result_1:.4f}")
    print(f"Sentence embeddings result on keywords of CV: {result_2:.4f}")

    applicants_df = pd.read_pickle('../text representation/applicant_ngrams.pkl')
    applicants_kw_df = pd.read_pickle('../text representation/applicant_kw_ngrams.pkl')
    companies_df = pd.read_pickle('../text representation/company_ngrams.pkl')

    applicant_result_df = get_top_matches(applicants_df, companies_df, 'ngrams')
    applicant_kw_result_df = get_top_matches(applicants_kw_df, companies_df, 'ngrams')

    result_1 = amount_of_matches(applicant_result_df)
    result_2 = amount_of_matches(applicant_kw_result_df)
    result_1 = result_1 / total
    result_2 = result_2 / total
    print(f"n-grams result on whole CV: {result_1:.4f}")
    print(f"n-grams result on keywords of CV: {result_2:.4f}")

def amount_of_matches(results_df):
    amount = 0
    for applicant_id, company_ids in ground_truth.items():
        if applicant_id in results_df['ID of Applicant'].values:
            result_row = results_df.loc[results_df['ID of Applicant'] == applicant_id]
            top_ids = result_row['Top 20 Similar IDs from Companies'].iloc[0]
            for top_id in top_ids:
                if top_id in company_ids:
                    amount += 1
    return amount

def get_top_matches(applicants_df, companies_df, option):
    applicant_embeddings = np.stack(applicants_df[option].values)
    company_embeddings = np.stack(companies_df[option].values)

    distance_matrix = manhattan_distances(applicant_embeddings, company_embeddings)

    top_20_indices = []
    applicant_ids = []

    for idx, distances in enumerate(distance_matrix):
        top_indices = np.argsort(distances)[:20]

        top_20_indices.append(companies_df['Submission ID'].iloc[top_indices].values.tolist())
        applicant_ids.append(applicants_df['Submission ID'].iloc[idx])

    results_df = pd.DataFrame({
        'ID of Applicant': applicant_ids,
        'Top 20 Similar IDs from Companies': top_20_indices
    })

    return results_df


if __name__ == "__main__":
    main()
