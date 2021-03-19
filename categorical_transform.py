from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import CategoricalDtype
import pandas as pd

class CategoricalTransform(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, min_data_portion = 0):
        self.cat_cols = cat_cols
        self.min_data_portion = min_data_portion
        
    def _transform_column(self, col, col_name):
        return col.astype(self.cat_type[col_name]) 
        
    def transform(self, df, **transform_params):
        df_cat = df.copy()
        for col in self.cat_cols:
            df_cat[col] = self._transform_column(df_cat[col], col)
        return df_cat
        
    def fit(self, X, y=None, **fit_params):
        self.cat_type = dict()
        for col in self.cat_cols:
            category_count = X.groupby(col).size().reset_index(name='count')
            filtered_category_count = category_count[category_count['count']>self.min_data_portion*len(X)]
            self.cat_type[col] = CategoricalDtype(filtered_category_count[col])
        return self
    

class IntegerCategoricalTransform(CategoricalTransform):
    def _transform_column(self, col, col_name):
        return super()._transform_column(col, col_name).values.codes

    
class OneHotTransform(BaseEstimator, TransformerMixin):
    def transform(self, df, **transform_params):
        return pd.get_dummies(df)
    
    def fit(self, X, y=None, **fit_params):
        return self
