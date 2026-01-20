from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_model():
    """Create TF-IDF + LogisticRegression pipeline"""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=5000
        )),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42))
    ])