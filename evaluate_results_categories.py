import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

max_matches = 0
industries = [
    "Administration, ekonomi, juridik","Bygg och anläggning","Chefer och verksamhetsledare",
    "Data/IT","Försäljning, inköp, marknadsföring","Hantverksyrken","Hotell, restaurang, storhushåll",
    "Hälso- och sjukvård","Industriell tillverkning","Installation, drift, underhåll","Kropps- och skönhetsvård",
    "Kultur, media, design","Militärt arbete","Naturbruk","Naturvetenskapligt arbete",
    "Pedagogiskt arbete","Sanering och renhållning",
    "Socialt arbete","Säkerhetsarbete","Tekniskt arbete","Transport"
]


def check_match(row_applicant, row_company):
    for industry in industries:
        titles_applicant = row_applicant[industry]
        if len(titles_applicant) != 0:
            titles_applicant_list = titles_applicant.split(", ")
        titles_company = row_company[industry]
        titles_company_list = titles_company.split(", ")

        for title in titles_applicant_list:
            if title in titles_company_list:
                print("True")
                return True
    print("False")
    return False

def main():

    applicants_df = pd.read_pickle('../text representation/applicant_sentence_embeddings.pkl')
    companies_df = pd.read_pickle('../text representation/company_sentence_embeddings.pkl')

    applicant_embeddings = np.stack(applicants_df['embeddings'].values)
    company_embeddings = np.stack(companies_df['embeddings'].values)

    similarity_matrix = cosine_similarity(applicant_embeddings, company_embeddings)

    top_20_indices = []
    top_20_scores = []
    applicant_ids = []

    # Find indices and scores of the top 20 similar companies for each applicant
    for idx, similarities in enumerate(similarity_matrix):
        top_indices = np.argsort(similarities)[-20:][::-1]  # Get indices of top 20 similarities
        top_scores = similarities[top_indices]  # Get the top 20 similarity scores

        top_20_indices.append(companies_df['Submission ID'].iloc[top_indices].values.tolist())
        top_20_scores.append(top_scores.tolist())
        applicant_ids.append(applicants_df['Submission ID'].iloc[idx])

    results_df = pd.DataFrame({
        'ID of Applicant': applicant_ids,
        'Top 20 Similar IDs from Companies': top_20_indices,
        'Similarity Scores': top_20_scores
    })

    total_matches = get_total_matches(applicants_df, companies_df)

    print(total_matches)

def get_total_matches(applicants_df, companies_df):
    amount = 0
    for index_applicant, row_applicant in applicants_df.iterrows():
        for index_company, row_company in companies_df.iterrows():
            if check_match(row_applicant, row_company):
                print("did stuff")
                amount+=1

    return amount


if __name__ == "__main__":
    main()