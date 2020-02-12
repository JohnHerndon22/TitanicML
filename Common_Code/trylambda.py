#trylambda.py
import pandas as pd

def extract_title(df):
    finder = ', '
    
    for index, Name in df.iterrows():
        commaloc = Name[0].find(finder)+2
        endoftitleloc = len(Name[0]) - Name[0].find(' ', commaloc)
        df.loc[index,'Title'] = Name[0][commaloc:-endoftitleloc]
        
    return df

def extract_title2(Name):
    finder = ', '
    commaloc = Name.find(finder)+2
    endoftitleloc = len(Name) - Name.find(' ', commaloc)
    Title = Name[commaloc:-endoftitleloc]
        
    return Title

def extract_family_name(Name):
    # finder = ', '
    # commaloc = Name.find(', ')
    return Name[0:Name.find(', ')]
    
def maxer(Name, b):
    return max(len(Name), len(b))


df = pd.DataFrame({'Name':['Herndon, Mrs Carol','Nickels, Miss Cathy','Herndon, Mr Mike','Jones, Mrs Carol','Billington, Master Bruce'], 'b' : ['1','2','3.5','4','5']})

# print(df['Name'][3].find('Carol'))
# quit()


# c_df = df.copy()
# df = extract_title(c_df)

# df['c'] = df.apply(lambda x: maxer(x['Name'], x['b']), axis=1)
df['Title'] = df.apply(lambda x: extract_title2(x['Name']), axis=1)
# df['Title'] = df.apply(lambda x: max(len(x['a']), len(x['b'])), axis=1)
df['Family Name'] = df.apply(lambda x: extract_family_name(x['Name']), axis=1)

print(df)

