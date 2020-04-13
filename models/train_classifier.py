# import libraries
import sys
import pandas as pd
import sqlite3
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

# download necessary NLTK and scikit-learn data
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """Loads data from SQLite database and returns target variables.
    Arguments:
     - database_filepath: Path where the SQLite database is stored
    Returns:
     - X, Y: Target variables
     - category_names: Names of the target variables"""
    
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('Messages', engine)
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """Tokenizes text, normalizes the case, lemmatizes it and removes white
    spaces. Arguments:
     - text: text to be tokenize"""
    
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    
    # Initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Iterate through each token
    clean_tokens = []
    for tok in tokens:
        
    # Lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok.lower().strip())
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Builds ML model for classifying test. Uses a pipeline for tokenizing text, then performs a
    TF-IDF transformation and uses Random Forest classifier for multiple targets. Also, it performs a
    GridSearch to optimizing the parameters of the classifier."""
    
    # Build the ML pipeline
    pipeline = Pipeline([    
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10, min_samples_leaf = 1, \
                                                             max_features = 'auto', n_jobs = -1)))])
    
    # Establish the parameters to perform the GridSearch
    parameters = {
        'clf__estimator__n_estimators': [20],
        #'clf__estimator__min_samples_leaf': [10, 20],
        #'clf__estimator__max_features': [0.5, 1]
    }
    
    # Perform a GridSearch for optimizing the parameters of the model
    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model and outputs a classification report to assess its performance.
    Arguments:
     - model: ML model to be evaluated
     - X_test, Y_test: Test data to evaluate the model with
     - category_names: Names of the target variables to be evaluated"""
    
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_test, Y_pred, target_names = Y_test.keys()))


def save_model(model, model_filepath):
    """Saves ML model to a pickle file. Arguments:
     - model: ML model to be saved
     - model_filepath: Path where the model will be saved"""
    
    with open((str(model_filepath)), 'wb') as model_filepath:
        pickle.dump(model, model_filepath)


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