# prep titan file
import pandas as pd
import os


def female_upper_crust(pclass, sex):
    if sex == 1:
        if pclass < 3:
            return 1
        else:
            return 0
    else:
        return 0 
    
    return 0

def deck_breakdown(cabin):

    if cabin[:1] == 'T': 
        return 1
    elif cabin[:1] == 'G':
        return 2
    elif cabin[:1] == 'F':
        return 3
    elif cabin[:1] == 'E':
        return 4
    elif cabin[:1] == 'D':
        return 5
    elif cabin[:1] == 'C':
        return 6
    elif cabin[:1] == 'B':
        return 7
    elif cabin[:1] == 'A':
        return 8
    else:
        return 0


def reg_fare(Fare):

    if Fare < 15: 
        regFare = 1
    elif (Fare >= 15 and Fare < 30):
        regFare = 2
    elif (Fare >= 30 and Fare < 55):
        regFare = 3
    else:
        regFare = 4

    return regFare
    

def compute_family_size(SibSp, Parch):
    # Creating new familysize column
    familysize = SibSp + Parch +1 # the one if for the self
    familysize = str(familysize)
    
    return familysize

def determine_wealth(fare):
    if fare > 70:
        return 1
    else:
        return 0

def extract_title(Name):
    # Convert to high minded title
    title =  Name[Name.find(', ')+2:-(len(Name) - Name.find('. ', Name.find(', ')+2))]
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir', 'Dr']:
        return '1'
    elif title in ['Countess', 'Mme', 'the Countess','Lady', 'Dona']:
        return '1'
    elif title in ['Mlle', 'Ms', 'Mrs', 'Mr']:
        return '0'
    else:
        return '0'  

def convert_embarked_to_int(embarked):
    if embarked == "Q": 
        Nembarked = 1
    elif embarked == "C":
        Nembarked = 2
    else:
        Nembarked = 3
    return Nembarked

def convert_sex_to_int(Sex):
    if Sex == "male": 
        NSex = 0
    else:
        NSex = 1
    return NSex

# clear the screen 
os.system('clear')

# consonants
# fare_filler = -99999
# age_filler = -99999

# open the training dataset - prepare dataframe
train_data = pd.read_csv('train.csv')

avg_fare = train_data['Fare'].mean()
avg_age = train_data['Age'].mean()

# replace NaN values with fillers
train_data = train_data.fillna({
'Age':avg_age,
'SibSp':0,
'Parch':0,
'Fare':avg_fare,
'Cabin': ''})

train_data['familysize'] = train_data.apply(lambda x: compute_family_size(x['SibSp'], x['Parch']), axis=1)
train_data['Title'] = train_data.apply(lambda x: extract_title(x['Name']), axis=1)
train_data['Wealthy'] = train_data.apply(lambda x: determine_wealth(x['Fare']), axis=1)
train_data['Nembarked'] = train_data.apply(lambda x: convert_embarked_to_int(x['Embarked']), axis=1)
train_data['Nsex'] = train_data.apply(lambda x: convert_sex_to_int(x['Sex']), axis=1)
train_data['regFare'] = train_data.apply(lambda x: reg_fare(x['Fare']), axis=1)
train_data['female_uc'] = train_data.apply(lambda x: female_upper_crust(x['Pclass'], x['Nsex']), axis=1)
train_data['Deck'] = train_data.apply(lambda x: deck_breakdown(x['Cabin']), axis=1)



print(train_data.head())

train_data.to_csv('prep_train.csv', header=False)
train_data.to_csv('prep_train_head.csv', header=True)


# print information for review
print(train_data.columns)
print(train_data.describe())
print(train_data.head())

# prepare test data frame
test_data = pd.read_csv('test.csv')

avg_fare = test_data['Fare'].mean()
avg_age = test_data['Age'].mean()

test_data = test_data.fillna({
'Age':avg_age,
'SibSp':0,
'Parch':0,
'Fare':avg_fare,
'Cabin': ''})

test_data['familysize'] = test_data.apply(lambda x: compute_family_size(x['SibSp'], x['Parch']), axis=1)
test_data['Title'] = test_data.apply(lambda x: extract_title(x['Name']), axis=1)
test_data['Wealthy'] = test_data.apply(lambda x: determine_wealth(x['Fare']), axis=1)
test_data['Nembarked'] = test_data.apply(lambda x: convert_embarked_to_int(x['Embarked']), axis=1)
test_data['Nsex'] = test_data.apply(lambda x: convert_sex_to_int(x['Sex']), axis=1)
test_data['regFare'] = test_data.apply(lambda x: reg_fare(x['Fare']), axis=1)
test_data['female_uc'] = test_data.apply(lambda x: female_upper_crust(x['Pclass'], x['Nsex']), axis=1)
test_data['Deck'] = test_data.apply(lambda x: deck_breakdown(x['Cabin']), axis=1)


print(test_data.head())

test_data.to_csv('prep_test.csv', header=False)
test_data.to_csv('prep_test_head.csv', header=True)
