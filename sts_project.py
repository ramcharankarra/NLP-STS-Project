import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# -----------------------------
# 1. LOAD DATASET
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("data/stsbenchmark.tsv", sep="\t", on_bad_lines='skip', quoting=3)

# Keep only required columns
df = df[['sentence1', 'sentence2', 'score']].dropna()

# OPTIONAL: Use subset for faster execution (remove if not needed)
df = df.head(1000)

# Normalize scores (0 to 1)
df['score'] = df['score'] / 5.0

# -----------------------------
# 2. PREPROCESSING
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

df['sentence1'] = df['sentence1'].apply(clean_text)
df['sentence2'] = df['sentence2'].apply(clean_text)

print("Preprocessing completed.")

# -----------------------------
# 3. TF-IDF MODEL
# -----------------------------
def tfidf_similarity(s1, s2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([s1, s2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

print("Calculating TF-IDF similarity...")
tfidf_scores = [tfidf_similarity(df['sentence1'][i], df['sentence2'][i]) for i in range(len(df))]
df['tfidf_score'] = tfidf_scores

# -----------------------------
# 4. BERT MODEL
# -----------------------------
print("Loading BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def bert_similarity(s1, s2):
    embeddings = model.encode([s1, s2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

print("Calculating BERT similarity...")
bert_scores = [bert_similarity(df['sentence1'][i], df['sentence2'][i]) for i in range(len(df))]
df['bert_score'] = bert_scores

# -----------------------------
# 5. HYBRID MODEL (INNOVATION)
# -----------------------------
df['hybrid_score'] = 0.3 * df['tfidf_score'] + 0.7 * df['bert_score']

# -----------------------------
# 6. EVALUATION
# -----------------------------
tfidf_corr = pearsonr(df['score'], df['tfidf_score'])[0]
bert_corr = pearsonr(df['score'], df['bert_score'])[0]
hybrid_corr = pearsonr(df['score'], df['hybrid_score'])[0]

print("\n========== FINAL RESULTS ==========")
print(f"TF-IDF Correlation   : {tfidf_corr:.4f}")
print(f"BERT Correlation     : {bert_corr:.4f}")
print(f"Hybrid Correlation   : {hybrid_corr:.4f}")

# -----------------------------
# 7. SAVE RESULTS
# -----------------------------
print("\nSaving results to results/results.csv and graph.png...")
df.to_csv("results/results.csv", index=False)

# -----------------------------
# 8. VISUALIZATION
# -----------------------------
models = ['TF-IDF', 'BERT', 'Hybrid']
scores = [tfidf_corr, bert_corr, hybrid_corr]

plt.figure()
plt.bar(models, scores)
plt.title("Model Performance Comparison")
plt.xlabel("Models")
plt.ylabel("Pearson Correlation")
plt.savefig("results/graph.png")
plt.show()

print("\nProject evaluation completed successfully!")

# -----------------------------
# 9. INTERACTIVE DEMO (LIVE INPUT)
# -----------------------------
print("\n--- Try it yourself! ---")
sentence1 = input("Enter first sentence: ")
sentence2 = input("Enter second sentence: ")

# Calculate live scores
live_tfidf = tfidf_similarity(sentence1, sentence2)
live_bert = bert_similarity(sentence1, sentence2)
live_hybrid = 0.3 * live_tfidf + 0.7 * live_bert

print(f"\nLive TF-IDF Score : {live_tfidf:.4f}")
print(f"Live BERT Score   : {live_bert:.4f}")
print(f"Live Hybrid Score : {live_hybrid:.4f}")
