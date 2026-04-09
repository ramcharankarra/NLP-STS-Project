# Semantic Textual Similarity using Hybrid NLP Model

## Introduction
Semantic Textual Similarity (STS) is an important task in Natural Language Processing that focuses on measuring how similar two sentences are based on their meaning. Unlike simple keyword matching, STS aims to capture the semantic relationship between sentences.

## Dataset
The dataset used in this project is the STS Benchmark dataset. It contains sentence pairs along with similarity scores given by human annotators. The dataset is referenced through resources associated with Princeton WordNet, which provides lexical and semantic relationships.

## Survey
Semantic similarity has been studied using different approaches over time. Early methods were based on lexical similarity, such as the Bag-of-Words model and TF-IDF, where similarity is calculated using term frequency and inverse document frequency. These approaches are simple but do not consider context.

To improve semantic understanding, knowledge-based methods using WordNet were introduced. These methods use relationships like synonymy and hypernymy, but they are limited for complex sentences.

With the development of deep learning, transformer-based models such as BERT have been introduced. BERT generates contextual embeddings and improves semantic understanding significantly.

## Research Gap
From the survey, it is observed that traditional methods like TF-IDF focus only on word-level similarity and fail to capture actual meaning. Deep learning models like BERT provide better understanding but are computationally expensive. Therefore, there is a need for a balanced approach combining both methods.

## Innovation
In this project, a hybrid model is proposed that combines TF-IDF and BERT similarity scores. TF-IDF captures lexical similarity, while BERT captures contextual meaning. A weighted combination of both is used to improve performance.

Hybrid Score = 0.3 × TF-IDF + 0.7 × BERT

This approach helps in utilizing both word-level and semantic-level information.

## Implementation
The implementation includes preprocessing, feature extraction, similarity computation, and evaluation.

- Text is cleaned by converting to lowercase and removing special characters  
- TF-IDF is applied to compute similarity based on word frequency  
- BERT is used to generate embeddings and compute semantic similarity  
- Both scores are combined to get the final hybrid score  

## Code
The project is implemented in Python using libraries such as pandas, scikit-learn, sentence-transformers, and matplotlib. The main steps include loading the dataset, preprocessing text, computing similarity using TF-IDF and BERT, and evaluating performance using Pearson correlation.

## Results and Analysis
The performance is evaluated using Pearson correlation.

- TF-IDF achieved a score of 0.4734  
- BERT achieved a score of 0.9165  
- Hybrid achieved a score of 0.8905  

From the results, it is clear that TF-IDF performs poorly because it does not understand context. BERT performs the best due to contextual embeddings. The hybrid model provides a balanced approach, although slightly lower than BERT due to the influence of TF-IDF.

## Conclusion
This project shows that combining traditional and deep learning approaches can improve robustness. While BERT gives the highest accuracy, the hybrid model provides a more balanced and practical solution.
