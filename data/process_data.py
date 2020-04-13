import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np


def load_data(messages_filepath, categories_filepath):
    """Loads both the messages and categories csv files,
    merges them and returns a dataframe. Arguments:
     - messages_filepath: path where the messages csv file is located
     - categories_filepath: path where the messages csv file is located
    Returns:
     - df: dataframe storing all data"""
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
    """Cleans the dataframe by separating the individual categories,
    and eliminating the duplicates and returns the clean dataframe.
    Arguments:
     - df: input dataframe
    Returns:
      - df: updated dataframe with clean data"""    
    
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat = ';', expand = True)
    
    row = categories.loc[0]

    # Use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[0:-2])

    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # Set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.extract('([0-9])')
    
    # Convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Drop the original categories column from `df`
    df = df.drop(columns = ['categories'])
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # Drop duplicates
    df = df[~df.duplicated()]
    df.related.replace(2, 1, inplace = True)
    return df
    

def save_data(df, database_filename):
    """Saves the dataframe into a SQLite database. Parameters:
     - df: dataframe storing all the data
     - database_filename: name of the SQLite database where the data will be stored"""
    
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('Messages', con = engine, index = False, if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()