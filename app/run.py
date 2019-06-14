import json
import plotly
import re
import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine
from collections import Counter


app = Flask(__name__)


def tokenize(text):
    """Tokenize each sentence.
    
        Args:
            text (string): twitter message
            
        Returns:
            list(tokens): list of tokens in the twitter message 
        
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatize
    lemmatizer = WordNetLemmatizer()

    stop_words = list(stopwords.words("english"))
    stop_words = stop_words + ["http", "com"]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) >= 3
    ]

    return tokens


# load data
engine = create_engine("sqlite:///data/disasters.db")
df = pd.read_sql_table("messages", engine)

# load model
model = pickle.load(open("models/classifier.pkl", "rb"))


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    """Generate Plots in the HTML master page.
    
        Args:
            None
            
        Returns:
            render_template(render_template): Render template for the plots  
        
    """
    # create visuals
    graphs = [
        bar_chart_messages_center(df),
        top_category_messages(df),
        histogram_categories_per_message(df),
        top_keywords_in_tweets(df),
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    """Generate a classification report.
    
        Args:
            None
            
        Returns:
            render_template(render_template): Render template for the classfication table
        
    """
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def bar_chart_messages_center(df):
    """Generate a bar chart for messages genre.
    
        Args:
            df (dataframe): twitter data stored in a dataframe
            
        Returns:
            dict: return a dictionary for the Bar plot data and layout 
        
    """
    genre_counts = df.groupby("genre").count()["message"].sort_values(ascending=False)
    genre_names = list(map(str.capitalize, list(genre_counts.index)))

    bar = Bar(
        x=genre_names,
        y=genre_counts,
        textposition="inside",
        hoverinfo="text",
        text=list(map(lambda x: "{} messages".format(x), genre_counts)),
        textfont=dict(color="black"),
        marker=dict(
            color="rgb(158,202,225)", line=dict(color="rgb(8,48,107)", width=1.5)
        ),
    )

    layout = {"title": "Distribution of Message Genres", "xaxis": {"title": "Genre"}}

    return {"data": [bar], "layout": layout}


def top_category_messages(df):
    """Generate a bar chart for the top 10 Categories.
    
        Args:
            df (dataframe): twitter data stored in a dataframe
            
        Returns:
            dict: return a dictionary for the Bar plot data and layout 
        
    """
    x = df.iloc[:, 3:].sum().sort_values(ascending=False).index.tolist()[:10]
    x = list(map(lambda x: x.replace("_", " ").capitalize(), x))
    y = df.iloc[:, 3:].sum().sort_values(ascending=False)[:10]

    bar = Bar(
        x=x,
        y=y,
        hoverinfo="text",
        textposition="inside",
        text=list(map(lambda x: "{}".format(x), y)),
        marker=dict(
            color="rgb(158,202,225)", line=dict(color="rgb(8,48,107)", width=1.5)
        ),
    )

    layout = {"title": "Top 10 Categoreis", "xaxis": {"title": "Category"}}

    return {"data": [bar], "layout": layout}


def histogram_categories_per_message(df):
    """Generate a histogram chart for the relative frequency of categories in each message.
    
        Args:
            df (dataframe): twitter data stored in a dataframe
            
        Returns:
            dict: return a dictionary for the histogram plot data and layout 
        
    """
    x = df.iloc[:, 3:].sum(axis=1).tolist()
    data = Histogram(
        x=x,
        histnorm="probability",
        marker=dict(
            color="rgb(158,202,225)", line=dict(color="rgb(8,48,107)", width=1.5)
        ),
    )

    layout = {
        "title": "Distribution of Categories Per Message",
        "yaxis": {"title": "Relative Frequency"},
        "xaxis": {"title": "Number of Categories Per Message"},
    }

    return {"data": [data], "layout": layout}


def top_keywords_in_tweets(df):
    """Generate a bar chart for the relative frequency for the most frequent keywords in the twitter data.
    
        Args:
            df (dataframe): twitter data stored in a dataframe
            
        Returns:
            dict: return a dictionary for the histogram plot data and layout 
        
    """
    # Top keywords in the Twitter data
    twitter_messages = " ".join(df[df["genre"] == "social"]["message"])
    twitter_tokens = tokenize(twitter_messages)
    twitter_wrd_counter = Counter(twitter_tokens).most_common()
    twitter_wrds = list(map(lambda x: x[0].capitalize(), twitter_wrd_counter))
    twitter_wrd_cnt = list(map(lambda x: x[1], twitter_wrd_counter))
    twitter_wrd_pct = list(
        map(lambda x: round(100 * x[1] / sum(twitter_wrd_cnt), 2), twitter_wrd_counter)
    )

    bar = Bar(
        x=twitter_wrds[:20],
        y=twitter_wrd_pct[:20],
        marker=dict(
            color="rgb(158,202,225)", line=dict(color="rgb(8,48,107)", width=1.5)
        ),
    )

    layout = {
        "title": "Top 20 Keywords in Social Media Messages",
        "xaxis": {"tickangle": 60},
        "yaxis": {"title": "Keyword Percentage"},
    }
    return {"data": [bar], "layout": layout}


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
