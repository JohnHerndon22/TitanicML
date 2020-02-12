#combinecsvs.py
import pandas as pd
from common import *

df_dt = pd.read_csv(titanDir+'t_decisiontree.csv', index_col=0)
print(df_dt.head)
df_rf = pd.read_csv(titanDir+'t_randomforest.csv', index_col=0)
df_ta = pd.read_csv(titanDir+'t_adaboost.csv', index_col=0)


df_final = pd.DataFrame()
df_final = df_rf
df_final.rename(columns = {'Survived':'RF_Survived'}, inplace = True)

for id, passenger in df_dt.iterrows(): df_final.loc[id,'DT_Survived'] = passenger.Survived
for id, passenger in df_ta.iterrows(): df_final.loc[id,'TA_Survived'] = passenger.Survived

print(df_final.head)




