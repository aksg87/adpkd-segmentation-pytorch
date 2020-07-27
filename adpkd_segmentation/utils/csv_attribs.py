# %%
import pandas as pd
from data.data_config import dataroot

# %%
csv_path = dataroot + "pkd_mutation_data/csv_attributes.csv"


class CsvAttribs:
    def __init__(self, path=None):
        super().__init__()

        if path == None:
            path = f"{dataroot}pkd_mutation_data/csv_attributes.csv"

        self.df = pd.read_csv(path)

    def __call__(self):

        return self.df

    def fig1_patient_summary(self, export=True):
        fig1 = self.df.copy()
        fig1["MRI_COUNT"] = fig1["ID"].str[-1:]
        fig1["ID"] = fig1["ID"].str[-9:-3]

        fig1 = fig1.drop(
            [
                "PKD_MUTATION",
                "PKD_TRUNCATING",
                "EGFR_AT_SCAN",
                "EGFR_AT_SCAN_DATE",
                "LATEST_EGFR_DATE",
                "TIME_EGFR_AT_SCAN_to_LATEST",
            ],
            axis=1,
        )

        cols = list(fig1.columns)
        cols = cols[-1:] + cols[:-1]
        fig1 = fig1[cols]

        fig1 = fig1.dropna()

        fig1.drop_duplicates(["ID"], keep="last", inplace=True)

        if export:
            fig1.to_csv("fig_1_patient_summary.csv", index=False)

        return fig1
