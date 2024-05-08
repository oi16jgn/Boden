import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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

    for index, row in results_df.iterrows():
        print(f"Applicant ID: {row['ID of Applicant']}")
        print("Top 20 Matches:")
        for company_id, score in zip(row['Top 20 Similar IDs from Companies'], row['Similarity Scores']):
            print(f"\t{score:.4f}, {company_id}")
        print("-" * 20)

    for index, row in results_df.iterrows():
        print(f"{row['Top 20 Similar IDs from Companies']}")

if __name__ == "__main__":
    main()
