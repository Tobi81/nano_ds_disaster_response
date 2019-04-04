import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/Disaster.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/Disaster.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    cats = df.loc[df['related'] >= 1.0]
    cats_counts = []
    cats_names = []
    for c in df.iloc[:,5:].columns:
        cats_counts.append(cats[c].sum())
        cats_names.append(c)
    cats_counts, cats_names = (list(t) for t in zip(*sorted(zip(cats_counts, cats_names))))

    social = df.loc[df['related'] >= 1.0]
    social2 = df.loc[df['genre'] == 'social']
    soc_cats_counts = []
    soc_cats_names = []
    for c in social2.iloc[:,5:].columns:
        soc_cats_counts.append(social2[c].sum())
        soc_cats_names.append(c)
    soc_cats_counts, soc_cats_names = (list(t) for t in zip(*sorted(zip(soc_cats_counts, soc_cats_names))))
    
    # create visuals
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
                    x=cats_names,
                    y=cats_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            "data": [
                {
                    "type": "pie",
                    "labels": soc_cats_names,
                    "values": soc_cats_counts
                }
            ],
            "layout": {
                "title": "Distribution of Social Media Messages",
                "width": 1000,
                "height": 1000
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()