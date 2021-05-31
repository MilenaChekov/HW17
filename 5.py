import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_pickle("bc_data.pkl")
ann = pd.read_pickle("bc_ann.pkl")

train = df_tr = df.loc[ann.loc[ann["Dataset type"] == "Training"].index]
test = df.loc[ann.loc[ann["Dataset type"] == "Validation"].index]

ttest=[ttest_ind(train[gene], test[gene])[1] for gene in df.columns]
print(pd.DataFrame(ttest, df.columns))

