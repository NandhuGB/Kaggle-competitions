
# Specialisation for Normalisation
class Normalisation:
    def normalise(self, df_column):
        raise NotImplementedError
    

# Min Max normalisation
from sklearn.preprocessing import MinMaxScaler
class MinMaxNormalisation(Normalisation):
    """
    For data with fixed upper and lower bounds. scalled data ranged from 0 to 1.
    k-nn, svm, neural nerworks
    """
    def normalise(self, df_column):
        min_max_scaler = MinMaxScaler()
        data_normalised = min_max_scaler.fit_transform(df_column)
        return data_normalised
    

# Z-score Normalisation
from sklearn.preprocessing import StandardScaler
class ZScoreNormalisation(Normalisation):
    """
    for data with normal distribution: return data with mean 0 and standard deviation 1
    logistic regression and linear regression (assumption - data is normal distributed)
    """
    def normalise(self, df_column):
        z_score_scaler = StandardScaler()
        data_standarised = z_score_scaler.fit_transform(df_column)
        return data_standarised
    
# Robust Scalling
from sklearn.preprocessing import RobustScaler

class RobustScallingNormalisation(Normalisation):
    """
    data with extreme outliers.
    
    """
    def normalise(self, df_column):
        robust_scaler = RobustScaler()
        data_scaled = robust_scaler.fit_transform(df_column)
        return data_scaled
