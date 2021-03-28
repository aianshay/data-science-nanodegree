import json
import plotly
import pandas as pd
import numpy as np
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from joblib import load
from sqlalchemy import create_engine


app = Flask(__name__)

def process_text(text):
    sub_text = re.sub('[^A-Za-z0-9]', ' ', text)
    tokens = word_tokenize(sub_text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = load("../models/classifier.joblib")


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')
    

@app.route('/classify')
def classify():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template('classify.html',
                            query=query,
                            classification_result=classification_results) 


@app.route('/data')
def data():

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categories = df.drop(columns=['id', 'message', 'original', 'genre'])
    categories_cols = list(categories.columns)
    categories_values = categories.sum().values
    cat = pd.DataFrame(columns=['category', 'count'])
    cat['category'] = categories_cols
    cat['count'] = categories_values
    cat.sort_values(by='count', ascending=False, inplace=True)

    tokens_list = [process_text(message) for message in df['message']]
    corpus = []

    for lista in tokens_list:
        for token in lista:
            corpus.append(token)

    d = dict()
    for word in corpus:
        if word in d:
            d[word] = d[word] + 1
        else:
            d[word] = 1

    words = pd.DataFrame(columns=['word', 'count'])
    words['word'] = list(d.keys())
    words['count'] = [d[word] for word in d.keys()]

    words.sort_values(by='count', ascending=False, inplace=True)
    stop_words = stopwords.words('english')
    words = words.query('word not in @stop_words')[:10]

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=np.flip(cat['count'][:10]),
                    y=np.flip(cat['category'][:10]),
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Most frequent messages categories',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=np.flip(words['count']),
                    y=np.flip(words['word']),
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Most frequent words',
                'yaxis': {
                    'title': "Word"
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('data.html', ids=ids, graphJSON=graphJSON)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()

