import sys
import numpy as np
import pandas as pd

import sqlite3
from sqlalchemy import create_engine

import pickle

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    """
    The function loads the database and extracts the messages and categories.
    INPUT : the database file path
    OUTPUT : 
        - the disaster messages as a series object, 
        - the categories associated with the messages as a dataframe object.
        - the categories names.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("select * from disaster_messages", engine)

    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    
    return X,Y,category_names


def tokenize(text):
    """
    The function tokenizes and lemmatizes the text in argument, then removes
    any characters not alphanumerical, then converts the tokens into lower characters.
    Afterwards, it removes english stop words.
    The function finally returns a list of tokens.
    INPUT : the text to process.
    OUTPUT : the tokens. 
    """
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]
       
    return tokens


def build_model():
    """
    The function builds the model with parameters.
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer(smooth_idf=False)),
    ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    
    parameters = dict(clf__estimator__max_depth=[2,4])
    
    cv = GridSearchCV(pipeline,param_grid=parameters)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """
    The function evaluates and scores the model.
    INPUT: 
            - the model object name
            - X_test data as series object
            - Y_test data as a dataframe object
    OUTPUT : the accuracy metrics
    """
    Y_pred = model.predict(X_test)
    
    for column_index, column_name in enumerate(category_names):
        print(column_name, '\n',classification_report(Y_test[column_name], Y_pred[:, column_index]))
    accuracy_per_feature = (Y_pred == Y_test).mean()
    mean_accuracy = np.mean(accuracy_per_feature)
    print("Final mean accuracy : {:.5f}".format(mean_accuracy))
    



def save_model(model, model_filepath):
    """
    The function saves the model into a pickle file.
    INPUT : the model name and the file path of the model.
    OUTPUT : the pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()