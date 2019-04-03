import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads data from messages and categories csv files.

    Args:
    messages_filepath: filepath to the messages csv file
    categories_filepath: filepath to the categories csv file

    Returns:
    df: DataFrame with merged messages and categories data
    categories: DataFrame containing only the categories data 
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='outer', on=['id'])
    return df, categories


def clean_data(df, categories):
    """Cleans the data: expands category column and drops duplicates.

    Args:
    df: DataFrame with merged messages and categories data
    categories: DataFrame containing only the categories data

    Returns:
    df: DataFrame with cleaned messages and categories data
    """
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(pat=";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    category_colnames = []
    for col in categories.columns:
        splitted = row[col].split('-')
        category_colnames.append(splitted[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split(pat="-", expand=True)[1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df = df.drop(labels=['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1, sort=False)
    
    # drop duplicates
    df.drop_duplicates(subset=['message'], inplace=True)
    return df


def save_data(df, database_filename):
    """Saves DataFrame to sqlite database.

    Args:
    df: DataFrame with cleaned messages and categories data
    database_filename: filepath to the sqlite database file 
    """
    # create SQLAlchemy engine
    engine = create_engine('sqlite:///' + database_filename)
    
    # save DataFrame to sqlite
    df.to_sql('InsertTableName', engine, index=False) 
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)
        
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