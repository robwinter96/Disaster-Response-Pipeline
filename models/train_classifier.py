# import libraries
import sys

import nltk
import pandas as pd

from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import re
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from sklearn.linear_model import SGDClassifier

nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    '''
    Load data from messages and categories files
    
    INPUTS:
    database_filepath - File path to database that was created using 'process_data.py'
    
    OUTPUTS:
    X - Variable columns/array
    y - target/predictor column/array
    df - full dataframe from sql database    
    
    '''
    # load data from database
    engine = create_engine('sqlite:///data/TwitterData.db')
    df = pd.read_sql_table(database_filepath, engine)
    X = df.message.values
    y = df.iloc[:,3:].values
    return X, y, df


def tokenize(text):
    '''
    Normalize case and remove punctuation
    
    INPUTS:
    text - text data for a single message in the dataset
    
    OUTPUTS:
    clean_tokens - tokenized text
    
    '''
    
    text = re.sub(r"[^0-9a-zA-Z]"," ",text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords.words("english"):
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build a ML model. Utilise Pipeline to automate the pre-processing data task and then use Gridsearch to optimise the ML model (In this case SGDClassifier).
    
    The parameters of the model can be changed by editing the 'parameters' dictionary.
    
    INPUTS:
    None
    
    OUTPUTS:
    cv - optimised ML model using SGDClassifier
    
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, stop_words='english',
                                ngram_range=(1,1),
                                max_df=0.5
                                )),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier()))])

    parameters = {
        #'vect__ngram_range': ((1,1),(1,2)),
        #'vect__max_df' : (0.5,1.0),
        #'vect__max_features': (None, 5000),
        'tfidf__use_idf' : (True, False),
        #'tfidf__norm': ('l1','l2'),
        #'clf__estimator__alpha' :(0.00001, 0.000001)
    }
    
    print("\n\t[INFO] Creating pipeline using parameters - %s" %parameters)
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2.1, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, df):
    '''
    Evaluate and display the results of the optimised model
    
    INPUTS:
    model - optimised model
    X_test - varibale columns/array
    Y_test - target/predictor array/column
    df - dataframe that holds X_test and Y_test
    
    OUTPUTS:
    print statements
    
    '''
    y_pred = model.predict(X_test)
    accuracy = (y_pred == Y_test).mean()
    print("Accuracy:", accuracy)
   
    report_dict={}
    for col in range(0,y_pred[0].shape[0]):
        print('\n','\t\t','-'*10, df.columns[3:][col],'-'*10,'\n',
          classification_report(np.hstack(Y_test[:, col]), 
                                np.hstack(y_pred[:, col])))
    
    return 

def save_model(model, model_filepath):
    '''
    Save the ML model as a pickle file
    
    INPUTS:
    model - optimised model
    model_path - filepath to the save location of the pickle file
    
    '''    
    model_pickled_object = pickle.dump(model, open(model_filepath, 'wb'))
    return 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, df = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, df)

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