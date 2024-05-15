## Installera alla requirements
* Se till att vara i mappen 'Boden'
* pip install -r requirements.txt 

## En map angränsande till mappen 'Boden' med namn 'data' krävs innehållande .csv filer för företag och arbetare
### Ska vara kompatibel med dessa rader i koden:
* os.path.join('..', 'data', 'arbete-cv.csv')
* os.path.join('..', 'data', 'arbetsgivare.csv')
### Liknand e behövs en map 'text representation' angränsande också
### Ska vara kompatibel med liknande rader i koden:
* os.path.join('..', 'text representation', 'applicant_ngrams.pkl')
* Endast mappen behöver finnas


## Kör create_text_representations.py
* python create_text_representations.py

## Kör evalueringsprogram
### Manuell evaluering
* python check_similarities.py
* python similaritys_ngrams.py

### Evaluerning mot ground truth
* python ground_truth_evaluation_manhattan.py
* python ground_truth_evaluation_cosine.py

### Evaluerning mot kategoriell matchning (psuedokod finns för enklare förståelse i evaluate_results_using_categories.txt)
* python evaluate_categories_manhattan.py
* python evaluate_categories_cosine.py