import pandas as pd

df = pd.read_csv('img_classess_big.csv')

'''

class0                                                     3180
class1                                                     1265
class2                                                     1515
class3                                                     1275
class4                                                     1325
class5                                                     4443
class6                                                     3449
class7                                                        0
class8                                                     2240
class9                                                      131
class10                                                       0
class11                                                     657
class12                                                     461
class13                                                    2301
class14                                                     193
class15                                                     512
class16                                                     310
class17                                                       0
class18                                                       0
class19                                                    4882
class20                                                       0
class21                                                     591
class22                                                    1112
class23                                                       0
class24                                                       0
class25                                                       0
class26                                                       7
class27                                                       0
class28                                                       0
class29                                                     770
class30                                                      64
class31                                                     273
class32                                                    2629
class33                                                      90
class34                                                       0
class35                                                       0
class36                                                       0
class37                                                       0
class38                                                       0
class39                                                       0

'''

print(df)

# Calculate the sum of each column
column_sums = df.iloc[:, 2:].sum()

# Create a boolean mask for columns whose sum is smaller than T
mask = column_sums < 800

# Drop columns based on the mask
df = df.drop(df.columns[2:][mask], axis=1)

column_sums = df.iloc[:, 2:].sum()
# Create a boolean mask for columns whose sum is bigger than T
mask = column_sums > 4000

# Drop columns based on the mask
df = df.drop(df.columns[2:][mask], axis=1)

df.to_csv('img_classes_big_edited2.csv')