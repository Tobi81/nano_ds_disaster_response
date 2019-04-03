# Disaster Response Pipeline Project

### Summary:
In this project, disaster data from Figure Eight are analyzed to build a model for an API that classifies disaster messages.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### What's included:
Within the download you'll find the following directories and files, logically grouping common assets. You'll see something like this:

* app
	* template
		* master.html  # main page of web app
		* go.html  # classification result page of web app
	* run.py  # Flask file that runs app

* data
	* disaster_categories.csv  # data to process 
	* disaster_messages.csv  # data to process
	* process_data.py # ETL pipeline
	* Disaster.db   # database to save clean data to

* models
	* train_classifier.py # ML pipeline
	* Disaster.pkl  # saved model 

* README.md