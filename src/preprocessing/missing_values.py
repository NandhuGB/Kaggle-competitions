
# Strategy design for handling missing values
class MissingValuesStrategy:
    """
    Template class for handling missing values
    """
    def handle(self, df, column):
        """
        Method handles the missing value in given column from pandas dataframe

        Args:
            df (pandas.DataFrame): Pandas dataframe
            column : Column where missing values to be handled

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("Must implement handle method")
    

# Dropping missing values
class DroppingMissingValuesStrategy(MissingValuesStrategy):
    def handle(self, df, column):
        return df.dropna(subset = [column], inplace = True)


# Filling missing value with mean of that column
class FillMeanMissingValueStrategy(MissingValuesStrategy):
    def handle(self,df,column):
        df[column].fillna(df[column].mean(), inplace = True)
        return df


# Filling missing value with median of that column
class FillMedianMissingValuesStrategy(MissingValuesStrategy):
    def handle(self, df, column):
        df.fillna({column:df[column].median()}, inplace = True)
        return df


# Filling missing value with mode value of that column
class FillModeMissingValuesStrategy(MissingValuesStrategy):
    def handle(self, df, column):
        df[column].fillna(df[column].mode()[0], inplace = True)
        return df


# Filling missing value with constant value
class FillConstMissingValuesStrategy(MissingValuesStrategy):
    def __init__(self, const):
        self.const = const
    def handle(self, df, column):
        df.fillna({column:self.const}, inplace =True)
        return df


# Filling missing value with propagation method 
class PropagateMissingValuesStrategy(MissingValuesStrategy):
    def __init__(self, method):
        """
        Args:
            method (str): there is two propagation types supported
                            'ffill' - forward fill
                            'bfill' - backward fill
        """ 
        self.method = method
    def handle(self, df, column):
        df[column].fillna(method=self.method, inplace = True)
        return df