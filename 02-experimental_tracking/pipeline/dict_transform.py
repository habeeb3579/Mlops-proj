from sklearn.base import BaseEstimator, TransformerMixin

class DictTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to convert selected DataFrame columns to a dictionary
    format suitable for DictVectorizer.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.to_dict(orient='records')