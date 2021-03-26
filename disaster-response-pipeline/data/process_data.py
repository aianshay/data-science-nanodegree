import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_path, categories_filepath):
    """Loads data from .csv files.
    
    Parameters:
    -----------
    messages_path : str
        file path of the messages .csv file

    categories_filepath : str
        file path of the categories .csv file

    Returns 
    ------- 
    merged_df : DataFrame
        messages and categories data merged in one dataframe
    """
    
    messages_df = pd.read_csv(messages_path)
    categories_df = pd.read_csv(categories_filepath)
    merged_df = messages_df.merge(categories_df, how='inner', on='id')
    
    return merged_df


def clean_data(df):
    """Cleans data and creates categories columns on dataframe.
       Raw category column is in format: 
         "related-1;request-0;offer-0;aid_related-0;"
       In this case, the new columns will be:
         ['related', 'request', 'offer', 'aid_related']
        And values: 
         [1, 0, 0, 0]

    Parameters
    ----------
    df : DataFrame
        dataframe to be cleaned

    Returns
    -------
    clean_df : DataFrame
        cleaned and ready to use dataframe
    """

    #create 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    #select first row
    row = categories.iloc[0, :]
    #get columns names
    category_cols = row.str.split('-', expand=True)[0]
    categories.columns = category_cols

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [value[-1] for value in categories[column]]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df = df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1) 

    #remove duplicates
    clean_df = df.drop_duplicates()

    return clean_df


def save_data(df, database_filename):
    """Saves dataframe as a sqlite database file."""

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_path, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_path, categories_filepath))
        df = load_data(messages_path, categories_filepath)

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