from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MultivariateAnalysis:
    def analyse(self, df:pd.DataFrame):

        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    def generate_correlation_heatmap(self, df:pd.DataFrame):
        pass

    def generate_pairplot(self, df:pd.DataFrame):
        pass


class SimpltMultivariateAnalysis(MultivariateAnalysis):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        return super().generate_correlation_heatmap(df)

