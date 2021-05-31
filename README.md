## Data Science Portfolio

This repository contains all the projects necessary to complete the Data Scientist Nanodegree @ Udacity.

## Projects

### Insights from AirBnB data from Rio de Janeiro

Using data from [Inside Airbnb](http://insideairbnb.com/) containing every Airbnb listing in the city of Rio de Janeiro, I did an exploratory data analysis and aimed to answer three questions of interest:

- Which are the cheapest months to go?
- Where to stay?
- How to write the title of your first accommodation

After that, I also built a model to predict the price of an accommodation and found the most important features for pricing.
Everything is synthesized in a blog post [here](https://aian.me/2021/03/17/ds-post.html). If you wanna check the more technical part a notebook is available in the `eda-predicting-airbnb-prices` folder above.

### Disaster Response Pipeline

The goal of this project was to deploy a machine learning model that classifies a message in a web page. This envolved building an ETL pipeline and training a model, I used a dataset provided by Figure Eight, which contains thousands of tweets found during natural disasters. The steps involved were: 

- Data extraction
- Data cleaning/preprocessing
- Data storing in a SQLite database
- Training a Random Forest classifier
- Deploying the model on a webpage


### Recommendations with IBM

In this project, I created different kinds of recommendations engines for the users of the IBM Watson Studio platform, it makes recommendations about new articles it thinks they will like. The dataset was provided by IBM, which contains interactions between the users and the articles. The building of such algorithms can be found in the notebook above, it is divided as the following:

- Exploratory data analysis
- Data cleaning/preprocessing
- Rank-based Recommendations
- User-based Collaborative Filtering
- Content-based Recommendations
- Matrix Factorization with SVD

### Churn Prediction with Spark

In this project, I used the dataset of Sparkify, a fictitious music streaming service, containing every user interaction inside the app. With this data I could build a Random Forest model that classifies if a user churned or not. More interestingly, the most powerful features for predicting a churn were number of active days in the app and the number of thumbs down a user has given. I blogged about the whole proccess [here](https://aian.me/2021/03/17/ds-post.html).

### Gaussian Distributions

I also built a Python package that implements[Gaussian](https://en.wikipedia.org/wiki/Normal_distribution) and [Binomial](https://en.wikipedia.org/wiki/Binomial_distribution) distributions. The repository can be found [here](https://github.com/aianshay/gaussian-binomial-dists) 