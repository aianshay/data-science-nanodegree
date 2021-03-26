import sys
from sqlalchemy import create_engine
import pandas as pd
import re
from joblib import dump

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data(database_path):
    """Loads data from the database.
    
    Parameters
    ----------
    database_path : str
        database path

    Returns
    -------
    X : DataFrame
        features to feed ML model
    y : DataFrame
        labels to feed ML model
    categories : list
        every possible category 
    """

    engine = create_engine(f'sqlite:///{database_path}')
    df = pd.read_sql_table('messages', engine)

    X = df['message']
    y = df.drop(columns=['message', 'original', 'genre', 'id']) 

    categories = y.columns

    nltk.download('punkt')
    nltk.download('wordnet')

    return X, y, categories


def process_text(text):
    "Processes text removing puctuations, lowering case and lemmatizing."

    #remove punctuations
    sub_text = re.sub('[^A-Za-z0-9]', ' ', text)

    tokens = word_tokenize(sub_text.lower().strip())

    lemmatizer = WordNetLemmatizer()
    tokenized_text = [lemmatizer.lemmatize(token) for token in tokens]

    return tokenized_text


def build_model():
    """Returns a ML pipeline with vectorizer and model."""

    vectorizer = TfidfVectorizer(tokenizer=process_text)
    clf = RandomForestClassifier()
    pipeline = Pipeline([('vect', vectorizer), ('clf', clf)])

    return pipeline


def evaluate_model(model, X_test, y_test, categories):
    """Prints models metrics scoring on each possible category."""

    preds = model.predict(X_test)

    for i, col in enumerate(categories):
        y_true = y_test.iloc[:, i]
        y_preds = preds[:, i]
        print('Report for category {}: \n {}'
                .format(col.upper(), classification_report(y_true, y_preds)))


def save_model(model, save_path):
    """Saves a trained model to specified path."""

    dump(model, save_path)


def main():
    if len(sys.argv) == 3:
        database_path, save_path = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_path))
        
        X, y, categories = load_data(database_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, categories)

        print('Saving model...\n    MODEL: {}'.format(save_path))
        save_model(model, save_path)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()