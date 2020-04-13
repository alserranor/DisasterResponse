# Disaster Response Pipeline Project

Web app that classifies messages related to disasters so they can be forwarded to the corresponding agencies.

For this project, a dataset provided by [FigureEight](https://www.figure-eight.com/) was used to build and train the Machine Learning model to classify the messages.

# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run the ETL pipeline that cleans data and stores it in a SQLite database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run the ML pipeline that trains the classifier and saves the model:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app:
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

# Contents

- app/: All the resources used for the designed of the web app are described here

- data/: The original datasets from FigureEight and the script to clean and process the data (ETL pipeline) are included in this directory

- models/: The ML pipeline model and the script that builds it can be found here

# License
The code used in this repository is licensed under a MIT license included in LICENSE.txt

# Acknowledgments
Must give credit to Udacity, and FigureEight for providing the dataset.