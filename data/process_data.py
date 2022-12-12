import sys
# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

# merge 2 dataframes together 
def merge_data(msg, cats):
    '''
    Merge 2 dataframes together 
    
    INPUTS:
    msg - dataframe with messages data
    cats - dataframe with categories data
    
    OUTPUTS:
    df - a merged dataframe of msg and cats
    
    '''
    return msg.merge(cats, on='id').set_index('id')


def create_cat_cols(df):
    '''
    Create categorical columns for the dataframe
    
    INPUTS:
    df - normal df 
    
    OUTPUTS:
    df - Dataframe with categorical columns
    
    
    '''
    categories = df.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = pd.unique([r[0] for r in row.str.split('-')])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    return categories

# convert category to binary data
def cat_cols_to_binary(df):
    '''
    Turn category columns from the given df to binary values 
    
    INPUTS:
    df - normal categorical columns dataframe
    
    OUTPUTS:
    df - binary categorical columns dataframe
    
    
    '''
    for column in df:
        # set each value to be the last character of the string
        df[column] = [row[-1] for row in df[column]]

        # convert column from string to numeric
        df[column] = pd.to_numeric(df[column])
        
    df = df[df.related != 2]
    return df

def replace_cat_cols(df, cat_df):
    '''
    Replace the categrory columns from df to the ones found in cat_df
    
    INPUTS:
    df - dataframe
    
    OUTPUTS:
    cat_df - dataframe with categorical column
    
    '''
    # drop the original categories column from `df`
    df = df.drop(columns='categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(cat_df)
    
    return df

# Remove duplicates
def remove_duplicates(df):
    '''
    Remove duplicates from dataframe
    
    INPUTS:
    df - dataframe
    
    OUTPUTS:
    df - dataframe with duplicate data removed
    
    '''
    # drop duplicates
    df = df.drop_duplicates()
    df = df.drop(df[df.columns[(df == 0).all()]].columns, axis=1)
    
    return df

# Overall workflow
def clean_workflow(msg_csv, cat_csv, db):
    '''
    Full workflow of recieving the csv data for messages and categories and store the cleaned data into a SQL database.
    
    INPUTS:
    msg_csv - File path to messages csv file
    cat_csv -  File path to categories csv file
    db - Filename to give to the database
    
    OUTPUTS:
    None
    
    '''    
    engine = create_engine(db)

    # Load data
    messages, categories = load_data(msg_csv, cat_csv)
    
    # clean and merge messages and categories df
    df = merge_data(messages, categories)
    
    # create category columns and assign new df to categories
    categories = create_cat_cols(df)
    
    # convert the categories columns in categories to 0 and 1's
    categories = cat_cols_to_binary(categories)
    
    # replace the categories columns in df with the binary columns from categories
    df = replace_cat_cols(df, categories)
    
    # Remove duplicate rows 
    df = remove_duplicates(df)
    
    # add to sql database
    df.to_sql(db, engine, index=False, if_exists='replace')
    
    return df
    
# Load data
def load_data(messages_filepath, categories_filepath):
    '''
    Load data from messages and categories files
    
    INPUTS:
    messages_filepath - File path to messages csv file
    categories_filepath - File path to categories csv file
    
    OUTPUTS:
    df - merged categories and messages csv data into a dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = merge_data(messages, categories)
    
    return df

# clean data              
def clean_data(df):
    '''
    Clean the dataframe object that is passed. 
    
    INPUTS:
    df - current dataframe
    
    OUTPUTS:
    df - updated df with replaced category columns
    
    '''
    cat_df = create_cat_cols(df)
    cat_df = cat_cols_to_binary(cat_df)
    df = replace_cat_cols(df, cat_df)
    df = remove_duplicates(df)
    return df

def save_data(df, database_filename):
    '''
    Save the df into a sql database using SQLAlchemy
    
    INPUTS:
    df - A clean and merged dataframe of emergency response data
    database_filename - the filename to give to the database
    
    OUTPUTS:
    Add data to SQL database / None
    '''
    engine = create_engine('sqlite:///data/TwitterData.db')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')
    return 


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
