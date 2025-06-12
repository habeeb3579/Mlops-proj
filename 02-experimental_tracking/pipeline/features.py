import pandas as pd

class FeatureProcessor:
    """Class for feature engineering operations"""
    
    def __init__(self):
        self.categorical_features = ['PU_DO']
        self.numerical_features = ['trip_distance']
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create necessary features from raw dataframe
        
        Args:
            df: Raw dataframe with taxi trip data
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        df = df.copy()
        # Create PU_DO combined feature
        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
        return df
