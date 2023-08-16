import pandas as pd

df = pd.read_csv('img_classes_big_edited.csv')

train_df = df[:3435]
test_df = df[3435:4171]
val_df = df[4171:]

#print(train_df.sum())
##print(test_df.sum())
#print(val_df.sum())

print(df.sum())