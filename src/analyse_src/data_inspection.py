from abc import ABC, abstractmethod
import pandas as pd
class DataInspectionStrategy(ABC):
    def inspect(self, df:pd.DataFrame):
        """
        Abstract method for all inheriting classes
        """
        raise NotImplementedError
    

class DataTypeInspectionStrategy(DataInspectionStrategy):

    def inspect(self, df: pd.DataFrame):
        """
        Method prints the data types of features in the given data type

        parameters (df:pd.DataFrame) :  Pandas Data frame which needed to be analysed

        side Effect: prints the data types of the given features
        return: None
        """

        print("\nDataTypes and Non-Null counts:")
        print(df.info())

        return None


class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):

    def inspect(self, df: pd.DataFrame):
        """
        Method prints the data types of features in the given data type

        parameters (df:pd.DataFrame) :  Pandas Data frame which needed to be analysed

        side Effect: prints the data types of the given features

        return: None
        """

        print(f"Descriptive Statistics of Numerical Datatypes:")
        print(df.describe(exclude=["O"]))
        print(f"Summary Statistics of Categorical Datatypes:")
        print(df.describe(include=["O"]))
        return None

