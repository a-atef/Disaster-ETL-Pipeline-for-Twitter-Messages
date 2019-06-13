# Disaster Response ETL Pipeline for Twitter Data

by: Ahmed A. Youssef

## Data Source

The data was obtained by Figure Eight Inc [insideairbnb website](https://www.figure-eight.com)

## Libraries Dependency
- re
- sklearn
- pandas
- plotly
- flask
- heroku
- nltk
- sqlalchemy
- pickle

## Project Motivation

- Extract, transfer, and load Twitter data obtained by figure eight into a SQLite database. 
- Construct a Machine Learning Pipeline to classify messages into 36 categories.
- Deploy the ETL pipeline and the ML classier to a console app.
- Build a dashboard to visualize the twitter data and generate a web report for classifying images.

## Project Files

- ```data\process_data.py```: module for reading, cleaning, transforming, and loading the twitter data into a SQLite database.
- ```models\train_classifier.py```: module to train and evaluate a multi-output classifier on the twitter data.
- ```app\run.py```: module to deploy and run the flask app into a local server. The same module can be used to build a Heroku app, as well. 
- ```app\templates\go.html```: display the classification report for a message. 
- ```app\templates\master.html```: display four plots to show the characteristics of the training dataset. 

## Analysis Workflow

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run an ETL pipeline to clean, transform and store the new data in a SQLite database
        ```python data/process_data.py -m data/disaster_messages.csv -c data/disaster_categories.csv -d data/disasters```
    - To run a pipeline for mutli-ouput classifier
        ```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```

2. Run the following command in the app's directory to deploy and fire a local web app using flask server.
    - ```python run.py```
    - Go to http://localhost:3001/

3. Run the following command to generate a Heroku app on your local machine: 
    -````heroku local web -f Procfile.windows````

Note: For more information on deploying a flask app or a Heroku app for the first time, refer to these links:
- http://www.gtlambert.com/blog/deploy-flask-app-to-heroku
- http://www.learningaboutelectronics.com/Articles/How-to-run-heroku-locally-on-a-windows-operating-system-Django.php
- https://devcenter.heroku.com/articles/python-nltk
- https://devcenter.heroku.com/articles/getting-started-with-python

### Results
After firing the web app, you should be able to see a dashboard with four plotly figures:
- Bar chart providing an overview of the genres available for each Twitter message in the training dataset.

<img src="img\Overview of Training Dataset.png"
     alt="Overview of Training Dataset"
     height="60%" width="60%"
     style="margin: 10px;" />

- Bar chart showing the top 10 categories in all twitter messages in the training dataset.

<img src="img\Top 10 Categories.png"
     alt="Top 10 Categories"
     height="60%" width="60%"
     style="margin: 10px;" />

- Histogram showing the distribution for the number of categories in each message.

<img src="img\Categories Per Message.png"
     alt="Categories Per Message"
     height="60%" width="60%"
     style="margin: 10px;" />

- Bar chart showing the top 10 keywords in the Twitter messages.

<img src="img\Top keywords.png"
     alt="Top keywords"
     height="60%" width="60%"
     style="margin: 10px;" />

- You can classify a tweet by writing it in the search bar and pressing the classify button. This will transfer you to a page with a classification report that will show the category of this tweet message. 

<img src="img\Classify.png"
     alt="Classify a message"
     height="60%" width="60%"
     style="margin: 10px;" />


<img src="img\Classify Report.png"
     alt="Classification Report"
     height="60%" width="60%"
     style="margin: 10px;" />
