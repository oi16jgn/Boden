import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def main():
    applicants_df = pd.read_pickle('../text representation/applicant_sentence_embeddings.pkl')
    companies_df = pd.read_pickle('../text representation/company_sentence_embeddings.pkl')

    # Convert the embeddings in the DataFrames from lists (or any other format) to numpy arrays
    applicant_embeddings = np.stack(applicants_df['embeddings'].values)
    company_embeddings = np.stack(companies_df['embeddings'].values)

    similarity_matrix = cosine_similarity(applicant_embeddings, company_embeddings)

    top_5_indices = []
    top_5_scores = []
    applicant_ids = []

    # Find indices and scores of the top 5 similar companies for each applicant
    for idx, similarities in enumerate(similarity_matrix):
        top_indices = np.argsort(similarities)[-10:][::-1]  # Get indices of top 10 similarities
        top_scores = similarities[top_indices]  # Get the top 5 similarity scores

        # Store results
        top_5_indices.append(companies_df['id'].iloc[top_indices].values.tolist())
        top_5_scores.append(top_scores.tolist())
        applicant_ids.append(applicants_df['id'].iloc[idx])

    # Create results DataFrame
    results_df = pd.DataFrame({
        'ID of Applicant': applicant_ids,
        'Top 5 Similar IDs from Companies': top_5_indices,
        'Similarity Scores': top_5_scores
    })

    for index, row in results_df.iterrows():
        print(f"Applicant ID: {row['ID of Applicant']}")
        print("Top 5 Matches:")
        for company_id, score in zip(row['Top 5 Similar IDs from Companies'], row['Similarity Scores']):
            print(f"\t{score:.4f}, {company_id}")
        print("-" * 40)


if __name__ == "__main__":
    main()
