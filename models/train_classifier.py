import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier #0.4 0.28 0.28 16500
from sklearn.linear_model import RidgeClassifierCV #
from sklearn.tree import DecisionTreeClassifier #0.34 0.34 0.34 16300
from sklearn.ensemble import ExtraTreesClassifier #0.39 0.28 0.28 16100
from sklearn.neural_network import MLPClassifier #
from sklearn.neighbors import KNeighborsClassifier #0.39 0.28 0.27 16600

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sqlalchemy import create_engine


def load_data(database_filepath):
    """Loads data from sqlite database file.

    Args:
    database_filename: filepath to the sqlite database file 
    
    Returns:
    X: DataFrame with feature variable (message column)
    Y: DataFrame with all target variables
    """
    
    # read in data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    
    # drop columns not needed
    df = df.drop(labels=['id', 'original', 'genre'], axis=1)
    
    # drop rows with NaN values (within target variables)
    df = df.dropna()
    
    # define feature variables
    X = df.iloc[:,0] 
    
    # define target variables, convert to integer and replace some incorrect values in related column
    Y = df.iloc[:,1:]
    Y = Y.astype('int64')
    Y['related'] = Y['related'].replace(2, 1)
    
    return X,Y


def tokenize(text):
    """Normalizes, tokenizes and lemmatizes a given text.

    Args:
    text: given raw text 
    
    Returns:
    clean_tokens: normalized, tokenized and lemmatized text tokens
    """
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # normalize and lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """Builds the ML model that has a MultiOutputClassifier at the end of the pipeline.

    Returns:
    cv: built model (grid search applied to ML pipeline)
    """
    
    # set up pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    
    #print(pipeline.get_params())
    
    # define parameters for grid search
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 500, 1000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__min_samples_leaf': (1, 2, 4),
        'clf__estimator__min_samples_split': (2, 4, 6)
    }

    # apply grid search to pipeline
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """Predicts test set and outputs a classification report.

    Args:
    model: the trained ML model
    X_test: test set feature values
    Y_test: test set target values
    """
    
    # predict test set target values
    Y_pred = model.predict(X_test)
    
    # print classification report
    print(classification_report(Y_test, Y_pred, target_names=Y_test.columns))


def save_model(model, model_filepath):
    """Saves ML model to a pickle file.

    Args:
    model: the trained ML model
    model_filepath: the filepath the model is saved to
    """
    
    # save model to a pickle file
    with open(model_filepath, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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