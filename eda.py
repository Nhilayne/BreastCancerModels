import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer


PATH = "/datasets/"
CSV = "haberman.csv"
dataset = pd.read_csv(PATH + CSV, index_col=None)

# show all columns
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(dataset.describe())
print(dataset.head())


transformer = Binarizer(threshold=1, copy=False).fit(dataset[['survival']])
awards_binary = pd.DataFrame(data=transformer.transform(dataset[['survival']]), columns=["survival_binary"])
new_dataset = pd.concat(([dataset, awards_binary]), axis=1)

transformer = Binarizer(threshold=4, copy=False).fit(dataset[['nodes']])
awards_binary = pd.DataFrame(data=transformer.transform(dataset[['nodes']]), columns=["nodes_over_4"])
new_dataset = pd.concat(([new_dataset, awards_binary]), axis=1)

transformer = Binarizer(threshold=40, copy=False).fit(dataset[['age']])
awards_binary = pd.DataFrame(data=transformer.transform(dataset[['age']]), columns=["age_over_40"])
new_dataset = pd.concat(([new_dataset, awards_binary]), axis=1)

print("\nsurvival data counts:\n", new_dataset['survival'].value_counts())

corr = new_dataset.corr()
print("\n\n",new_dataset.head(10))
# plot the heatmap
plt.figure(figsize=(10,10))
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            annot=True)

plt.show()
