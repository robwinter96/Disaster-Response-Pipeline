# Disaster Response Pipeline
 <i>By Rob Winter, 12/12/2022</i>

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The standard Anaconda distribution of Python should be sufficient to run the project files

The code was created and run using Python versions 3.* and higher

## Project Motivations <a name="motivation"></a>

### Introduction 

As part of the Udacity Data Science course, I have developed a NLP pipeline to classify emergency response data.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. 

## File Descriptions <a name="files"></a>

There are three components encompassed in this NLP pipeline. 

### ETL Pipeline
process_data.py contains a cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

### ML Pipeline
train_classifier.py writes a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file


### Flask Web App
A flask web app that:

- Displays 3 graphs that display different analysis of the dataset.
- Message classification tool

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Results <a name="results"></a>

The ML pipeline I have created has successfully cleans and adds data to a sql database. It then has the ability to classify messages.

The final classification is far from perfect and still has room for additional improvements.

## Licensing, Authors, and Acknowledgements <a name="licensing"></a>


The data source for this project was given by Appen and is part of the Udacity Data Scientist course
