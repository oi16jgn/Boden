import pandas as pd
import matplotlib.pyplot as plt
import preprocessing
from keybert import KeyBERT


def main():
    # applicants = preprocessing.prepare_applicant_data()
    # companies = preprocessing.prepare_company_data()

    kw_model = KeyBERT(model='paraphrase-multilingual-mpnet-base-v2')

    text = "Bodensläp tillverkar släpvagnar för tungtrafik, där huvudprodukten är timmersläp. Utöver detta utför vi även bilpåbyggnation samt utför service och reparationer på alla typer av tunga släp. Hos oss är det kunden som står i centrum och all tillverkning sker utifrån kundens behov och önskemål. Företagets vision är att tillhandahålla kundanpassade produkter av hög kvalité."

    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), use_mmr=True,
                                      diversity=0.7)

    print(keywords)


if __name__ == "__main__":
    main()
