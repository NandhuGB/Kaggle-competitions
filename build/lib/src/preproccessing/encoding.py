

# Specilisation class
class Encoding:
    """
    - **For Nominal Variables**: One-hot encoding, binary encoding, or hashing.
    - **For Ordinal Variables**: Label encoding, target encoding.
    - **High-Cardinality Variables**: Target encoding, frequency encoding, binary encoding, or hashing.
    - **For Tree-based Models (e.g., Decision Trees, Random Forests, Gradient Boosting)**: Target       encoding or label encoding often works better than one-hot encoding.
    - **For Linear Models (e.g., Logistic Regression)**: One-hot encoding is usually preferred. 
    """
    def encode(self, df, column):
        raise NotImplementedError
    


# Label Encoding
from sklearn.preprocessing import LabelEncoder
class LabelEncoding(Encoding):
    """
    Assigns a unique integer to each category value in the feature - best for ordinal values
        **Example**:
            - Category: `["Low", "Medium", "High"]`
            - Encoded: `[0, 1, 2]`
    """
    def __init__(self):
        self.encoder = LabelEncoder()

    def encode(self, df, column):
        df[column] = self.encoder.fit_transform(df[column])
        return df


# One-Hot encoding
from sklearn.preprocessing import OneHotEncoder
class OneHotEncoding(Encoding):
    """
    Converts each category in the feature into seperate binary column - best for nominal values
    """

    def __init__(self):
        self.encoder = OneHotEncoder()

    def encode(self, df, column):
        encoded_data = self.encoder.fit_transform(df[column])
        return encoded_data


# Target/Mean encoding
class TargetMeanEncoding(Encoding):
    """
    Replaces each category with the mean of the target variable for the category
    best for tree based models
    """
    def __init__(self, target):
        self.target = target
    def encode(self, df, column):
        df[column] = df.groupby(column)[self.target].transform("mean")
        return df



# Frequency count Encoding
class FrequencyCountEncoding(Encoding):
    """
    Replaces each category by the number of times it appears in the dataset.
    reduces the dimensionality --> high cardinality categorical features
    """

    def encode(self, df, column):
        df[column] = df[column].map(df[column].value_counts())
        return df
    

# Binary Encoding
import category_encoders as ce
class BinaryEncoding(Encoding):
    """ 
    Combines label encoding and binary transformation. the categories are converted into integers then, those integers converted into binary form.
    good for high cardinality features --> reduces the number of columns compared to onehot encoding
    """
    def encode(self, df, column):
        binary_encoder = ce.BinaryEncoder(cols=[column])
        df[column] = binary_encoder.fit_transform(df)
        return df
    

# Hashing Encoding
from sklearn.feature_extraction import FeatureHasher
class HashingEncoding(Encoding):
    """
    This Encoder applies hash function to map categories into numerical space. 
    usefull for high cardinality features and streaming data
    """
    def __init__(self):
        self.hasher = FeatureHasher(input_type="string")

    def encode(self, df, column):
        df[column] = self.hasher.transform(df[column].astype(str))
        return df

# Leave-one-out Encoding

class LeaveOneOutEncoding(Encoding):
    """
    Similar to target encoding but leaves out the current row when calculating the mean to reduce overfitting.
    High-cardinality features for linear models 
    """
    def __init__(self, target):
        self.target = target

    def encode(self, df, column):
        loo_encoder = ce.LeaveOneOutEncoder(cols=[column])
        df[column] = loo_encoder.transform(df, df[self.target])
        return df


        
