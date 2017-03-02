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
from keras.models import Sequential
from keras.layers import Dense, Activation

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def remove_unused_columns(df):
    '''
    Remove columns that are do not contain useful information.

    :param df: dataframe
    :return: dataframe
    '''
    return df.drop(['PassengerId', 'Ticket', 'Name', 'Cabin', 'Fare', 'Embarked'], axis=1)


def fill_age(df):
    '''
    For the rows with no age information fill the age with random numbers.
    :param df: dataframe
    :return: dataframe
    '''
    nan_age = df['Age'].isnull().sum()
    mean_train = df['Age'].mean()
    std_train = df['Age'].std()

    random_values = np.random.randint(mean_train-std_train, mean_train+std_train, size=nan_age)
    df['Age'].loc[np.isnan(df['Age'])] = random_values
    df['Age'] = df['Age'].astype(int)
    return df


def family(df):
    '''
    Determine if a passenger had a family present on the ship.
    :param df:
    :return:
    '''
    df['Family'] = df['Parch'] + df['SibSp']
    df['Family'].loc[df['Family'] > 0] = 1
    df['Family'].loc[df['Family'] == 0] = 0
    df = df.drop(['SibSp', 'Parch'], axis=1)
    return df


def get_person(passenger):
    '''
    Determine if a passenger was a child.
    :param passenger: tuple (age, sex)
    :return: sex or child
    '''
    age, sex = passenger
    return 'child' if age < 16 else sex


def person(df):
    '''
    Add hot-encoding columns for Child, Male, Female.
    :param df: dataframe
    :return: dataframe
    '''
    df['Person'] = df[['Age', 'Sex']].apply(get_person, axis=1)
    df.drop(['Sex'], axis=1, inplace=True)

    # Create dummy variables for person column
    person_dummies = pd.get_dummies(df['Person'])
    person_dummies.columns = ['Child', 'Female', 'Male']
    person_dummies.drop(['Male'], axis=1, inplace=True)

    df = df.join(person_dummies)
    df.drop('Person', axis=1, inplace=True)

    return df

def prepare_data(df):
    '''
    Process data for further analysis.
    :param df: dataframe
    :return: dataframe
    '''
    df = remove_unused_columns(df)
    df = fill_age(df)
    df = family(df)
    df = person(df)
    return df

def split_data(train_df, test_df):
    '''
    Split data into X_train, Y_train, and X_test
    :param train_df:
    :param test_df:
    :return: dataframe, dataframe, dataframe
    '''
    X_train = train_df.drop('Survived', axis=1)
    Y_train = train_df['Survived']
    X_test = test_df.copy()
    return X_train, Y_train, X_test


def logistic_regression(train_df, test_df):
    '''
    Use logistic regression to predict if a passenger survived.
    :param train_df: dataframe
    :param test_df: dataframe
    :return: float
    '''
    X_train, Y_train, X_test = split_data(train_df, test_df)
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    score = logreg.score(X_train, Y_train)
    print(score)
    return score

def support_vector_machines(train_df, test_df, C):
    '''
    Use SVM to predict if a passenger survived.
    :param train_df: dataframe
    :param test_df: dataframe
    :return: float
    '''
    X_train, Y_train, X_test = split_data(train_df, test_df)
    svc = SVC(C=C, kernel='rbf')
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    score = svc.score(X_train, Y_train)
    print(score)
    return score

def random_forest(train_df, test_df):
    '''
    Use random forest for prediction.
    :param train_df: dataframe
    :param test_df: dataframe
    :return: float
    '''
    X_train, Y_train, X_test = split_data(train_df, test_df)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    score = random_forest.score(X_train, Y_train)
    print(score)
    return score

def mlp(train_df, test_df, epochs=100):
    '''
    Use MLP to predict if a passenger survived.
    :param train_df:
    :param test_df:
    :return:
    '''
    X, Y, test = split_data(train_df, test_df)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.6)
    X_train = X_train.values
    Y_train = Y_train.values
    X_test = X_test.values
    Y_test = Y_test.values

    model = Sequential()
    model.add(Dense(128, input_dim=5, activation='relu'))
    model.add(Dense(128,  activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, nb_epoch=epochs, validation_data=(X_test, Y_test))
    model.save_weights('weights.h5')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy')
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.show()


if __name__ == '__main__':

    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')

    train_df = prepare_data(train_df)
    test_df = prepare_data(test_df)

    print(train_df.columns)
    print(test_df.columns)

    # for C in [0.01, 0.03, 0.1, 0.3, 1, 30, 100, 300, 1000]:
    # support_vector_machines(train_df, test_df, 1000)
    # random_forest(train_df, test_df)
    # mlp(train_df, test_df, epochs=2000)

    colormap = plt.cm.viridis
    plt.figure(figsize=(12,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
    sns.plt.show()