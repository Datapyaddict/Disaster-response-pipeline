# <u>Data Scientist nanodegree / Udacity project : Disaster response pipeline</u>
## Table of Contents
1. [Project info](#project-info)
2. [Repository files info](#repository-files-info)
3. [How to run the scripts](#how-to-run-the-scripts)
4. [Acknowledgement](#Acknowledgement)
***

### Project info

The purpose of this project is to implement an ETL pipeline and a machine learning (ML) pipeline in order to analyse disaster data from Appen (formally Figure 8) and build a machine learning model for an API that classifies disaster messages.

The dataset contains real messages sent during disaster events.
 
The  ETL pipeline is used to load the disaster messages in csv format into a SQLite database. The ML pipeline will then train and test the dataset and save the model in a pickle file.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the dataset.

The csv files consist of :
* `disaster_messages.csv` containing disaster messages per genre,
* `disaster_categories.csv` containing the different categories of the messages.


***
### Repository files info:


* `ETL Pipeline Preparation.ipynb` contains the different steps that load the csv files, clean the data and save them into a SQLite database.
* `ML Pipeline Preparation.ipynb` contains the steps that load the dataset, build,train, test and evaluate the model, and finally save it into a pickle file.
* `data/disaster_categories.csv` contains the categories associated with the disaster messages.
* `data/disaster_messages.csv` contains the disaster messages.
* `data/DisasterResponse.db` is the SQLite database containing the dataset.
* `data/process_data.py` is the python script that loads the csv files.
* `models/train_classifier.py` is the python script that builds the model, trains, tests the dataset before saving the model as a pickle file.
* `models/classifier.pkl` is the pickle file containing the ML model.
* `app/run.py` is the script that loads the database and uses the ML model in order to classify the messages that can be input through the web app. 


***

### How to run the scripts


1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database :
    
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves :
    
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app:    `python run.py`

3. Go to `//localhost:3001` to visualize the html page of the web app.


![web app](web-app.png)

*** 

### Acknowledgement
Thanks to Udacity for the starter code of the web app.
