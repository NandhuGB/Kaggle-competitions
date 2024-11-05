from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Base class for univariate analysis Strategy
# -------------------------------------------
# This class defines a common interface for univariate anlaysis strategies
# Subclasses must implement the analyse method
class UnivariateAnalysis(ABC):
    """ 
    Perform univariate analysis on a specific feature of the dataframe

    Parameters:
    df (pandas.DataFrame): Pandas dataframe 
    column: Feature to be analysed
    """
    @abstractmethod
    def analyse(self, df, column,mode:list):
        """
        """
        raise NotImplementedError


class NumericalUnivariateAnalysis(UnivariateAnalysis):
    def anlayse(self, df:pd.DataFrame, column:str, mode):
        plt.figure(figsize=(10,6))
        sns.histplot(data = df,x = column,kde=True, bins=30)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()


class CategoricalUnivariateAnalysis(UnivariateAnalysis):
    def analyse(self, df, column, mode):

        if "count" in mode or "all" in mode:
            
            plt.figure(figsize=(10,6))
            sns.countplot(x=column, data=df, palette = "muted")
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.show()

        if "pie" in mode or "all" in mode:

            plt.figure(figsize = (10,6))
            plt.pie(x=column, data=df)
            plt.title(f"Pie Chart of {column}")
            plt.xlabel(column)
            plt.ylabel("Pie")
            plt.show()

        if "bar" in mode or "all" in mode:
            
            plt.figure(figsize=(10,6))
            sns.barplot(data = df, x=column)
            plt.title(f"Barplot of {column}")
            plt.xlabel(column)
            plt.ylabel("Bar")
            plt.show()
