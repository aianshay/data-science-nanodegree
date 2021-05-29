## Data Science Portfolio

This repository contains all the projects necessary to complete the Data Scientist Nanodegree @ Udacity.

## Projects

### Insights from AirBnB data from Rio de Janeiro

Using data from Inside Airbnb, I did an exploratory data analysis of the dataset, then answered three questions:

- Which are the cheapest months to go?
- Where to stay?
- How to write the title of your first accommodation

After that, I also built a model to predict the prices of an accommodation and found the most important features for pricing.
Everything is synthesized in a blog post [here](https://aian.me/2021/03/17/ds-post.html). If you wanna check the more technical part a notebook is available in the `eda-predicting-airbnb-prices` folder above.

### Disaster Response Pipeline

The goal of this project was to deploy a machine learning model that classifies a message in a web page, this envolved building an ETL pipeline and training the model, I used a dataset provided by Figure Eight, which contains thousands of tweets found during a natural disaster. The steps involved were: 

- Data extraction
- Data cleaning/preprocessing
- Data storing in a SQLite database
- Training a Random Forest classifier
- Deploying the model on a webpage

On the ETL part I used pandas, sqlalchemy and SQLite database. To build the classifier I used sci-kit learn and to deploy the model I used bootstrap 5 on the frontend and Flask as the backend.

