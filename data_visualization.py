import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import numpy as np

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
# plt.plot(data)
# print(data)
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
print(train_df.columns)
print(train_df.Survived.sum()/len(train_df))

def factorplot(df, factor):
    g = sns.factorplot(factor, 'Survived', data=df, size=4, aspect=3)


def countplot(df, factor):
    fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(15, 5))

    embark_perc = df[[factor, 'Survived']].groupby([factor], as_index=False).mean()

    sns.countplot(x=factor, data=df, ax=axis1)
    sns.countplot(x='Survived', hue=factor, data=df, order=[1, 0], ax=axis2)
    sns.barplot(x=factor, y='Survived', data=embark_perc, hue=factor, ax=axis3)

    sns.plt.show()


def avg_age(df):
    # peaks for survived/not survived passengers by their age
    facet = sns.FacetGrid(df, hue="Survived", aspect=4)
    facet.map(sns.kdeplot, 'Age', shade=True)
    facet.set(xlim=(0, df['Age'].max()))
    facet.add_legend()

    # average survived passengers by age
    fig, axis1 = plt.subplots(1,1,figsize=(18,4))
    average_age = df[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()
    sns.barplot(x='Age', y='Survived', data=average_age)
    sns.plt.show()


def fill_age(df):
    nan_age = df['Age'].isnull().sum()
    mean_train = df['Age'].mean()
    std_train = df['Age'].std()

    random_values = np.random.randint(mean_train-std_train, mean_train+std_train, size=nan_age)
    df['Age'].loc[np.isnan(df['Age'])] = random_values
    df['Age'] = df['Age'].astype(int)
    return df

def family(df):
    df['Family'] = df['Parch'] + df['SibSp']
    df['Family'].loc[df['Family'] > 0] = 1
    df['Family'].loc[df['Family'] == 0] = 0

    df = df.drop(['SibSp', 'Parch'], axis=1)

    fig, (axis1, axis2) = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
    sns.countplot(x='Family', data=df, order=[1,0], ax=axis1)
    family_perc = df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
    sns.barplot(x='Family', y='Survived', data=family_perc, order=[1, 0], ax=axis2)
    axis1.set_xticklabels(['With Family', 'Alone'], rotation=0)
    sns.plt.show()

def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex

def person(df):
    df['Person'] = df[['Age', 'Sex']].apply(get_person, axis=1)
    df.drop(['Sex'], axis=1, inplace=True)

    try:
        fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
        sns.countplot(x='Person', data=df, ax=axis1)
        person_perc = df[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()
        sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male', 'female', 'child'])
        sns.plt.show()
    except:
        pass

    return df

def drop_columns(df):
    df.drop(['SibSp', 'Parch', 'Fare', 'Embarked'], axis=1, inplace=True)
    return df


train_df.info()
test_df.info()

# factorplot(train_df, 'Sex')
# countplot(train_df, 'SibSp')


train_df = fill_age(train_df)
test_df = fill_age(test_df)

train_df = person(train_df)
test_df = person(test_df)

# avg_age(train_df)
# family(train_df)

X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()

print(X_train.head(10))
print(Y_train.head(10))

# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
#
# logreg.score(X_train, Y_train)

# print(len(X_train), len(Y_train))

# sns.countplot(x='Embarked', data=train_df, ax=axis1)
# sns.countplot(x='Survived', hue="Embarked", data=train_df, order=[1, 0], ax=axis2)



