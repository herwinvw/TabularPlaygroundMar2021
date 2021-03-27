from sklearn.base import BaseEstimator, TransformerMixin # type: ignore
from pandas.api.types import CategoricalDtype # type: ignore
import pandas as pd # type: ignore
from typing import List, Any

class CategoricalTransform(BaseEstimator, TransformerMixin):
    """
    Transforms selected categorical columns in a CategoricalDtype, leaving the other columns as is.
    
    The categories used in the CategoricalDtypes are to be fitted from training data. At transform, 
    unknown categories are transformed to None.
    """
    def __init__(self, cat_cols:List[str], min_data_portion:int = 0):
        """
        Args:
            cat_cols: names of the columns to encode in a CategoricalDtype
            min_data_portion: minimum portion (0..1) that a category should have in the training set. Potential categories which contain less than min_data_portion samples are encoded as None.
        """
        self.cat_cols = cat_cols
        self.min_data_portion = min_data_portion
        
    def _transform_column(self, col:pd.Series, col_name:str)->pd.Series:
        return col.astype(self.cat_type[col_name]) 
        
    def transform(self, df:pd.DataFrame, **transform_params:Any)->pd.DataFrame:
        """
        Transforms df[cat_cols] into the CategoricalDtypes learned from training.
        Categories not seen in training, or with fewer than min_data_portion rows in training are encoded as None.
        
        Args:
            df: DataFrame to transform
            transform_params: unused
            
        Returns:
            The transformed DataFrame
        """
        df_cat = df.copy()
        for col in self.cat_cols:
            df_cat[col] = self._transform_column(df_cat[col], col)
        return df_cat
        
    def fit(self, X:pd.DataFrame, y:Any=None, **fit_params:Any):
        """
        Learns the CategoricalDtype for each categorical feature
        
        Args:
            X: the DataFrame to learn from
            y: unused
            fit_params: unused
        """
        self.cat_type = dict()
        for col in self.cat_cols:
            category_count = X.groupby(col).size().reset_index(name='count')
            filtered_category_count = category_count[category_count['count']>self.min_data_portion*len(X)]
            self.cat_type[col] = CategoricalDtype(filtered_category_count[col])
        return self
    

class IntegerCategoricalTransform(CategoricalTransform):
    """
    Transforms selected categorical columns in a int, leaving the other columns as is.
    
    The categories used are to be fitted from training data. At transform, 
    unknown categories and categories that contain less than min_data_portion rows in the fit
    are transformed to -1.
    """
    def _transform_column(self, col:pd.Series, col_name:str)->pd.Series:
        return super()._transform_column(col, col_name).values.codes

class NonNegativeIntegerCategoricalTransform(CategoricalTransform):
    """
    Transforms selected categorical columns in a non-negative int, leaving the other columns as is.
    
    The categories used are to be fitted from training data. At transform, 
    unknown categories and categories that contain less than min_data_portion rows in the fit
    are transformed to 0.
    """
    
    def _transform_column(self, col:pd.Series, col_name:str)->pd.Series:
        return super()._transform_column(col, col_name).values.codes+1
    
class OneHotTransform(BaseEstimator, TransformerMixin):
    """
    One hot encode all columns with type CategoricalDtype, leaving the other columns as is.
    """
    
    def transform(self, df:pd.DataFrame, **transform_params:Any)->pd.DataFrame:
        """
        One hot encode all columns with type CategoricalDtype, leaving the other columns as is.
    
        Args:
            df: DataFrame to transform
            transform_params: unused
        Returns:
            the one hot encoded DataFrame
        """
        return pd.get_dummies(df)
    
    def fit(self, X:pd.DataFrame, y:Any=None, **fit_params:Any):
        """
        Not used
        """    
        return self
