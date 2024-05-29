import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

columns = [
    'Administration, ekonomi, juridik',
    'Bygg och anläggning', 'Chefer och verksamhetsledare', 'Data/IT',
    'Försäljning, inköp, marknadsföring', 'Hantverksyrken',
    'Hotell, restaurang, storhushåll', 'Hälso- och sjukvård',
    'Industriell tillverkning', 'Installation, drift, underhåll',
    'Kropps- och skönhetsvård', 'Kultur, media, design', 'Militärt arbete',
    'Naturbruk', 'Naturvetenskapligt arbete', 'Pedagogiskt arbete',
    'Sanering och renhållning', 'Socialt arbete', 'Säkerhetsarbete',
    'Tekniskt arbete', 'Transport'
]

df = preprocessing.load_companies()

non_null_counts = df[columns].notnull().sum()

non_null_counts_sorted = non_null_counts.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
non_null_counts_sorted.plot(kind='bar')
plt.xlabel('Industries')
plt.ylabel('Amount')
plt.title('Amount of companies in industries')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
