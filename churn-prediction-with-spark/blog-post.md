# Churn Prediction with Spark

![Image](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/customer.png)

Churn is defined as the event when a user leaves or unsubscribes from a service. Predicting this event is already an important part of businesses as Netflix, Spotify and YouTube. When predicting this event, companies can offer incetives so the user doesn't leave the plataform, potentially saving a lot of money. Other than that, it's also an opportunity of understanding why users are leaving the product, and which improvements can be made.

In this case, I'll be using pySpark, a Python API for manipulating distributed datasets and creating machine learning models. With it, it's easier to handle datasets that don't easily fit into memory. 

## The Dataset

I'm gonna use a dataset provided by Udacity. It contains data from the Sparkify app, a fictional music streaming service. Each line of the dataset contains one user behavior, indicating which user made an action in the app and which page it was.

Let's get started!

## Exploratory Data Analysis

### Loading data

It's easy to load the data using the following command:

```python
df = spark.read.json('mini_sparkify_event_data.json')
```

I also need to create a temporary view, where I can use SQL querys to investigate the data. It's necessary to pass a name as argument:

```python
df.createOrReplaceTempView('sparkify_data')
```

First let's have a glimpse of the dataset:

![Dataset](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/data.png)

Indeed, each line indicates one user behavior, where it's possible to see **which page the user was at**, if he was listening to any music, gender, id of the session, user name etc. Now let's check all the columns and it's types.

![Columns](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/columns.png)

The number of lines can be easily obtained with the `count()` function: 

```python 
df.count()
```

That outputs the number of 286500 lines.

### Unique users

It would be interesting if we had the number of unique users:

```python
spark.sql("""select count (distinct userId) from sparkify_data""").show()
```

![Unique users](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/unique_users.png)


Let's check how frequent each user is in the dataset with the following query:

```python
spark.sql("""select userId, count(userId) as count
             from sparkify_data
             group by userId
             order by count desc""").show()
```

![Users count](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/users_count.png)

Looks like we got a Null value that happens 8346 times. I'll investigate this later.

### Page frequencies

Now let's figure out how frequent each type of page is:

```python
spark.sql("""select page, count(page) as count
             from sparkify_data
             group by page
             order by count desc""").toPandas()
```

![Pages](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/pages.png)

From this I can draw some conclusions:

- `NextSong` is by far the most frequent page, which makes sense in a streaming service.
- `Submit Registration` is the least frequent page.
- The column `Cancellation Confirmation` will be used to define churned users.
- There are 22 possible pages.
- With `page` column it's possible to create new features like number of friends, listening time, number of listened songs, number of thumbs up, etc. for each user. This will be done in the feature engineering step later on. 

### Free/paid users proportion

What about free/paid users?

```python
df.groupBy('level').count().show()
```

![Level](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/level.png)

There are almost 4x more paid users than free users. 


### Gender proportions

What about gender proportions?

```python
df.groupBy('gender').count().show()
```


<p align="center">
  <img src="https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/gender.png">
</p>

There are 8346 Null values in the gender column, the same value we found in the `userId` column. This probably refers to users that don't have an account in the app. These values will be dropped later in the feature engineering part.



## Feature Engineering

The feature engineering process will be composed of dropping the null values we found earlier and constructing the folowing features for each user:

- Listening time
- Active days
- Number of sessions
- Number of listened songs 
- Number of thumbs up
- Number of thumbs down
- Number of added friends
- Churn flag

Most of the features will be build using SQL commands.

### Dropping nulls

To drop users without id, we can use the Spark method `filter`:

```python
df.filter(df.userId != "")
```

### Listening time 

```python
listening_time = spark.sql("""select userId, sum(length) as listening_time  
                              from sparkify_data_churn
                              where page = 'NextSong'
                              group by userId""")
```

![Listening time](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/listening.png)


### Active days 

To calculate the number of days since registration, I first selected the `registration` column, which gives us when a user's account was created, and the most recent value of the column `ts`, which gives us the timestamp of each logged event in the app.

```python
active = spark.sql("""select userId, min(registration) as created, max(ts) as last_session 
                      from sparkify_data_churn
                      group by userId""")
```

Both columns are formatted as Unix timestamp so I had to create a method which converts these values and calculates the difference between them in days: 

```python
def compute_active_days(created, last_session):
    """Calculates the difference between an account date of registration and
        its most recent login date.
        
        Parameters:
        -----------
        created
            date of creation timestamp : str
        last_session
            last login date : str
            
        Returns:
        --------
        active_days : int
            number of active days
        """
    
    created = int(created)/1000
    last_session = int(last_session)/1000
    
    active_days = (datetime.fromtimestamp(last_session) - datetime.fromtimestamp(created)).days
    return active_days
```
To use the `compute_active_days` function above, I needed to wrap the method as a Spark User Defined Function, them I created a new column in the `active` dataframe which indicates number of active days since registration. 

```python
active_days = udf(compute_active_days, IntegerType())
active = active.withColumn("active_days", active_days(active.created, active.last_session))
```

![Active days](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/active_days.png)


### Number of sessions

```python
sessions = spark.sql("""select userId, count(distinct sessionId) as sessions  
                        from sparkify_data_churn
                        group by userId""")
```

![Sessions](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/sessions.png)

### Number of listened songs

```python
songs = spark.sql("""select userId, count(Song) as total_songs  
                     from sparkify_data_churn
                     group by userId""")
```

![Songs](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/songs.png)

### Number of thumbs up

```python
thumbs_up = spark.sql("""select userId, count(page) as thumbs_up  
                         from sparkify_data_churn
                         where page = 'Thumbs Up'
                         group by userId""")
```

![Thumbs up](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/thumbs_up.png)

### Number of thumbs down

```python
thumbs_down = spark.sql("""select userId, count(page) as thumbs_down 
                           from sparkify_data_churn
                           where page = 'Thumbs Down'
                           group by userId""")
```

![Thumbs down](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/thumbs_down.png)

### Number of added friends

```python
friends = spark.sql("""select userId, count(page) as friends  
                       from sparkify_data_churn
                       where page = 'Add Friend'
                       group by userId""")
```

![Friends](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/friends.png)

### Churn flag

Users who visited the page `Cancellation confirmation` will be assigned as churned. To do that, I defined another user defined function, I also created a new column to this flag:

```python
churn = udf(lambda x: 1 if x == 'Cancellation Confirmation' else 0)
df_2 = df_2.withColumn('churn_flag', churn(df.page))
```

Let's count the frequency of each flag:

```python
spark.sql("""select churn_flag, count(churn_flag)
          from sparkify_data_churn
          group by churn_flag""").show()
```

![Churn](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/churn.png)

From the image above, we can see that there are only 52 churns. As there are so few churns, in the modeling part, using F1 score for model evaluation is more suitable.

### Join

Now I have all the data necessary to train some models, but I first need to join all of the above datasets with the following script:

```python
user_data = listening_time.join(thumbs_up, on='userId', how='outer')\
                         .join(thumbs_down, on='userId', how='outer')\
                         .join(songs, on='userId', how='outer')\
                         .join(sessions, on='userId', how='outer')\
                         .join(friends, on='userId', how='outer')\
                         .join(active, on='userId', how='outer')\
                         .join(churn, on='userId', how='outer')
```

Then we get:

![Full data](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/full_data.png)

### Filling missing values

As we can see above, there is still some work to be done:
- I'll input NaNs with 0, as it makes sense that some users don't have friends or never used the thumbs up/down feature of the app.
- Columns `created` and `last_session` can be dropped.

Both steps can be made using the following script:

```python
full_df = user_data.drop('created').drop('last_session').fillna(0)
```

Which outputs as a result:

![Filled data](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/filled_data.png)

Yay! Now the data is finally ready to train some models.


### Scaling

But first, as many machine learning models expect data to be scaled, this can be done with the `StandardScaler` class on following script: 

```python
scaler = StandardScaler(inputCol = 'features', outputCol='scaled_features')
model_df = scaler.fit(model_df).transform(model_df)
```


## Modeling

As churn is a binary classification problem, I built three models avaiable on the pySpark API:

- Random Forest Classifier (RFC)
- Logistic Regression (LR)
- Gradient Boosted Trees (GBT)

The data was split in train and test with the `randomSplit()` method. 80% was used for training and 20% for testing:

```python
train, test = model_df.randomSplit([0.8, 0.2], seed = 42)
```

To score the models, I used the F1 score, as there are so few churned examples, with the class `MulticlassClassificationEvaluator`.

```python
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
```

### Results

The F1 score of each model was: 

| Model  | F1 Score       |  
| :-     |:-              |      
|  RFC   |      0.800     |
|  LR    |      0.663     | 
|  GBT   |      0.705     |      


Random Forest had the highest score, so I went futher to investigate its feature importances:


![Importances](https://raw.githubusercontent.com/aianshay/aianshay.github.io/master/_posts/images/importances.png)

Number of active days and number of sessions were the most important features for predicting churning. While listening time and number of thumbs down had a similar and significance. 

## Bonus

If you want to check out how I analyzed the data, a notebook is available [here][https://github.com/aianshay/data-science-portifolio/tree/main/churn-prediction-with-spark]. 

## Acknowledgments

Thanks to this Udacity for providing the dataset.