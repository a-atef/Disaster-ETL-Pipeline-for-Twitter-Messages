import sys
import nltk
import pickle
import re
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """Load database.
    
        Args:
            database_filepath (string): path to the SQLite database
            
        Returns:
            Dataframe (X): the input features for the ML algorithm
            Dataframe (Y): the target features for the ML algorithm 
            list (columns): list of output features
        
    """
    # load data from database
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table(con=engine, table_name="messages")
    X = df.message
    Y = df.drop(["message", "original", "genre"], axis=1)
    columns = Y.columns
    return X, Y, columns


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


def build_model():
    """Construct a ML model.
    
        Args:
            None
            
        Returns:
            GridSearch : GridSearch estimator 
        
    """

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(AdaBoostClassifier(random_state=42))),
        ]
    )

    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                    'tfidf__use_idf': [True],
                    'clf__estimator__n_estimators': [10, 50]
                    }

    return GridSearchCV(pipeline, param_grid=parameters, verbose=10, n_jobs=1)


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the performance of ML model.
    
        Args:
            model (GridSearch): GridSearch estimator 
            X_test (float): test inputs
            Y_test (bool): targeted test outputs 
            category_names (list): category names
            
        Returns:
            None 
        
    """
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=Y_test.columns)

    for column in category_names:
        print("Atrribute:{}\n".format(column))
        print(classification_report(Y_test[column], y_pred_df[column]))


def save_model(model, model_filepath):
    """Save the ML model to a pickle file.
    
        Args:
            model (GridSearch): GridSearch estimator 
            model_filepath (string): file path to save the pickle model
            
        Returns:
            Dataframe: new dataframe without rows or columns with NANs exceeding the threshold 
        
    """
    with open("{}.pkl".format(model_filepath), "wb") as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
