import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    the function loads the csv files disaster_messages.csv in argument messages_filepath,
    and disaster_categories.csv in argument categories_filepath.
    It merges the files into a dataframe and returns the dataframe.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how ='inner', on =['id'])
    
    return df

def clean_data(df):
    """
    The function breaks down the categories column content of the dataframe in argument into columns and assigns the numeric 
    values 0 or 1 to the columns values.
    When the numeric value is different from 0 and 1, it is replaced with value 2.
    Then the function removes duplicates and returns a clean dataframe
    """    
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0,:]
    category_column_names = row.apply(lambda x: x[:-2])
    categories.columns = category_column_names
    
    for column in category_column_names:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
        categories.replace(2,1, inplace = True)
    
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat((df, categories), axis=1)
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    The function saves the dataframe given in argument into an sqlite database 
    which filename path is given in 2nd argument. 
    OUTPUT : the table disaster_messages created in the SQLite databse.
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_messages', engine, if_exists='replace',index=False)  


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