# import needed packages
import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Read messages and categories data from csv files.
    
        Args:
            messages_filepath (string): path to the messages csv file 
            categories_filepath (string): path to the categories csv file
    
        Returns:
            Dataframe: new dataframe containing messages and categories data 
        
    """

    messages = pd.read_csv(messages_filepath, index_col="id")
    categories = pd.read_csv(categories_filepath, index_col="id")
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    """Transform and Clean the dataframe.
    
        Args:
            df (Dataframe): pandas dataframe containing messages and categories 
    
        Returns:
            Dataframe: clean dataframe ready for ML analysis 
        
    """

    categories = pd.DataFrame(df["categories"].str.split(";", expand=True))
    categories.columns = categories.iloc[0, :].str[0:-2].tolist()

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # extract a list of new column names for categories.
    category_colnames = row.str[0:-2].tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    categories = clean_outliers(categories)

    # drop the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df, categories, on="id")

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def clean_outliers(df):
    """Clean columns from outliers. Columns should have 1 or 0. For each column, outliers will be replaced by the most frequent value.
    
        Args:
            df (Dataframe): andas dataframe containing messages and categories 
    
        Returns:
            Dataframe: clean dataframe without outliers 
        
    """

    columns_with_outliers = df.apply(lambda x: x > 1).sum()
    columns_with_outliers = columns_with_outliers[
        columns_with_outliers > 0
    ].index.tolist()
    for col in columns_with_outliers:
        df[col] = df[col].replace({2: df[col].value_counts().index[0]})
    return df


def save_data(df, database_filename):
    """Save the clean dataframe into a SQLite database.
    
        Args:
            df (Dataframe): pandas dataframe containing the clean and transformed data about messages and categories
            database_filename (string): path where the SQLite database will be created and saved  
    
        Returns:
            None 
        
    """

    engine = create_engine("sqlite:///{}.db".format(database_filename))
    df.to_sql("messages", engine, index=False, if_exists="replace")


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--messages", required=True, help="path to messages data")
    ap.add_argument(
        "-c", "--categories", required=True, help="path to Caffe 'deploy' prototxt file"
    )
    ap.add_argument("-d", "--database", required=True, help="path to database file")
    args = vars(ap.parse_args())
    print(args)

    if len(args) == 3:
        messages_filepath, categories_filepath, database_filepath = (
            args["messages"],
            args["categories"],
            args["database"],
        )

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )

        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
