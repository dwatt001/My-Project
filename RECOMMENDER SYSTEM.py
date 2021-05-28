#!/usr/bin/env python
# coding: utf-8

# # DANIEL WATTS PROJECT - PYTHON POWERED MOVIE RECOMMENDATION ENGINE

# There are many types of movie recommendation systems, my one will be using three types of filtering.
# 
# Content Based Filtering- They suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.
# 
# Collaborative Filtering- This system matches persons with similar interests and provides recommendations based on this matching. Collaborative filters do not require item metadata like its content-based counterparts.
# 
# Demographic Filtering- They offer general recommendations to every user, based on movie popularity and/or genre. The System recommends the same movies to users with similar demographic features. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience.

# Loading the data/Data preprocessing

# MY FILE PATH IS DIFFERENT TO YOURS, PLEASE ENTER THE FILE NAME FOR THE CSV FILES FROM YOUR COMPUTER OR THE SYSTEM WILL NOT WORK

# In[1]:


import pandas as pd 
import numpy as np 

dataframe1=pd.read_csv('/Users/danielwatts/Desktop/Uni/Year 3/Computing Project/archive (1)/tmdb_5000_credits.csv')
dataframe2=pd.read_csv('/Users/danielwatts/Desktop/Uni/Year 3/Computing Project/archive (1)/tmdb_5000_movies.csv')


# This code conjoins the two datasets on the 'id' column

# In[2]:


dataframe1.columns = ['id','tittle','cast','crew']
dataframe2= dataframe2.merge(dataframe1,on='id')


# This line of code gives us a short preview of the data

# In[3]:


dataframe2.head(5)


# This line of code again gives us another little peak at the data

# In[4]:


dataframe2.describe()


# # Part One - Demographic Filtering

# For demographic filtering there are three metrics to be discovered.
# * A metric to score or rate movie
# * Calculating the score for every movie
# * Sorting the scores and recommending the best rated movie to the users.
# 
# In order to ensure integrity I will be using a weighted method because it would be inaccurate to give a movie a high rating if it only has a small amount like 7 votes. Therefore I will be using IMDB's weighted rating (wr) which is as follows.
# 
# Weighted rating  (𝑊𝑅)=(𝑣/(𝑣+𝑚))𝑅+(𝑚/(𝑣+𝑚))𝐶
# * R = Average rating of the movie
# * m = Minimum votes required to be listed in the chart
# * v = The number of votes for the movie
# * C = Mean amount of votes

# As v(vote_count) and R (vote_average) are already defined, The following code finds the value of C
# 

# In[5]:


C= dataframe2['vote_average'].mean()
C


# This has showed us that the approximate mean for the movies is just over 6. Now we need to determine a suitable amount for M which is a minimum value for the movie to be considered. I will use the 85th percentile as the cutoff, meanining that for a movie to be considered, it must have more votes than at least 85% of the other movies in the list.

# In[6]:


m= dataframe2['vote_count'].quantile(0.85)
m


# Now I can sift throught the movies and decide which ones qualify

# In[7]:


QualifiedMovies = dataframe2.copy().loc[dataframe2['vote_count'] >= m]
QualifiedMovies.shape


# This calculates that there are 721 movies that qualify to be in the list, now I need to calculate the metric for each qualified movie. So I will call define a weighted rating function which will define a new score, from this I will calculate the value by applying this function to my DataFrame of qualified movie

# In[8]:


def IMDB_Weighted_Rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# The code cell below creates a new value called score, which is the result of the IMDB weighted rating calculation

# In[9]:


# Defining a new feature 'score' and calculate its value with `IMDB_Weighted_Rating`
QualifiedMovies['score'] = QualifiedMovies.apply(IMDB_Weighted_Rating, axis=1)


# Now I am going to sort the DataFrame based on the score feature and output the title, vote count, vote average and weighted rating or score of the top 10 movies.

# In[10]:


#Sorting movies based on score calculated above
QualifiedMovies = QualifiedMovies.sort_values('score', ascending=False)

#Priningt the top 15 movies
QualifiedMovies[['title', 'vote_count', 'vote_average', 'score']].head(10)


# This is a basic implementation of a demographic based movie recommendation engine. This essentially finds the mean score and finds the most popular ones - very basic and doesn't take much attention to the finer points such as plot and content, which will come later

# The graph below depicts the movies with the highest vote counts, from this we can see that, Inception, The Dark Night and Interstellar all make it into the most popular movies in terms of weighted rating which is interesting to see.

# In[11]:


pop= dataframe2.sort_values('vote_count', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['vote_count'].head(6), align='center',
        color='darkblue')
plt.gca().invert_yaxis()
plt.xlabel("Vote counts")
plt.title("Highest Vote Counts")


# # Part Two - Content Based Filtering

# In this recommender system the content of the movie (overview, cast, crew, keyword, tagline etc) is used to find its similarity with other movies. Then the movies that are most likely to be similar are recommended. The first part of this system will use a pairwise similarity score based on descriptions and compute a similiarity score which will then generate a reccomendation.

# Pairwise similarity scores will be calculated for all movies based on their plot descriptions and recommend movies based on that similarity score. The plot description is given in the overview feature of our dataset.

# In[12]:


dataframe2['overview'].head(5)


# I will now compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each overview

# The following code was adapted from source:
# https://www.kaggle.com/aligsaoud/different-recommendation-systems
# Accessed 10/03/2021

# In[13]:


#Importing TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer. Removing all english words such as 'the', 'a'
TermFrequency_IDF = TfidfVectorizer(stop_words='english')

#Replacing NaN with an empty string
dataframe2['overview'] = dataframe2['overview'].fillna('')

#Constructing the required TF-IDF matrix by fitting and transforming the data
TermFrequency_IDF_matrix = TermFrequency_IDF.fit_transform(dataframe2['overview'])

#Outputting the shape of TermFrequency_IDF_matrix
TermFrequency_IDF_matrix.shape


# It's interesting to see that there is over 20,000 words used to describe the 4803 movies in our dataset.

# I will be using the cosine similarity rule that denotes similarities between two movies. The following code shows how the cosine similarity is computed, as you can see I make use of the sklearn library, which is absolutley critical for the content based filtering

# In[14]:


# Importing linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_similarity = linear_kernel(TermFrequency_IDF_matrix, TermFrequency_IDF_matrix)


# In[15]:


#Constructing a reverse map of indices and movie titles
indices = pd.Series(dataframe2.index, index=dataframe2['title']).drop_duplicates()


# The following code was adapted from source:
# https://www.kaggle.com/aligsaoud/different-recommendation-systems
# Accessed 10/03/2021

# In[16]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_similarity=cosine_similarity):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    Pairwise_similarity_scores = list(enumerate(cosine_similarity[idx]))

    # Sort the movies based on the similarity scores
    Pairwise_similarity_scores = sorted(Pairwise_similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    Pairwise_similarity_scores = Pairwise_similarity_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in Pairwise_similarity_scores]

    # Return the top 10 most similar movies
    return dataframe2['title'].iloc[movie_indices]


# ### Use this recommender to get films based from plot and descriptions

# Users - Insert movie name into red quote marks, use full capitals and relevant spacing

# In[17]:


get_recommendations('Star Wars')


# Users - Insert movie name into red quote marks, use full capitals and relevant spacing

# In[18]:


get_recommendations('The Departed')


# ### Metadata recommender - Content based filter

# I will now build a recommender system using key metadata such as top 3 actors, director and movie plot keywords

# In[19]:


from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    dataframe2[feature] = dataframe2[feature].apply(literal_eval)


# The following code cell was adapted from source https://www.kaggle.com/rounakbanik/movie-recommender-systems
# Accessed 12/03/2021

# In[20]:


# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# This code cell creates a get list function, essentially checking if more than 3 elements exist, if not then returning the entire list.

# In[21]:


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []


# This code cell gets directors, casts, keywords, and genres that are in a suitable format

# In[22]:


dataframe2['director'] = dataframe2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    dataframe2[feature] = dataframe2[feature].apply(get_list)


# This code cell prints the new features of the dataframe that will be examined in the metadata soup

# In[23]:


# Print the new Metadatafeatures of the first 3 films
dataframe2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)


# The next code cell converts the names and keyword instances into lowercase and strip all the spaces between them. This is done so that our vectorizer doesn't count the Daniel of "Daniel Craig" and " Daniel Radcliffe" as the same.

# In[24]:


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# The following is an example of data cleaning, as there is a lot of irrelevant fields which will just overcomplicate the calculations, I am notifying the system of whats important and highlighting this as clean data

# In[25]:


# Apply clean_data function to my features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    dataframe2[feature] = dataframe2[feature].apply(clean_data)


# The following cell creates my metadata soup which is a string that contains all metadata that I want to feed into the system.

# In[26]:


def create_Metadata_Soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
dataframe2['soup'] = dataframe2.apply(create_Metadata_Soup, axis=1)


# The next steps are the same as what I did with my plot description based recommender. One important difference is that we use the CountVectorizer() instead of TF-IDF. This is because I do not want to down-weight the presence of an actor/director if he or she has acted or directed in relatively more movies.

# In[27]:


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(dataframe2['soup'])


# The following cell is important as it defines my cosine similarity score, if we see my getrecommendations function then we can see that after the user input is "cosine_2" this is because we want to see the list of movies with the highest cosine similarity scores

# In[28]:


# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[29]:


# Reset index of our main DataFrame and construct reverse mapping as before
dataframe2 = dataframe2.reset_index()
indices = pd.Series(dataframe2.index, index=dataframe2['title'])


# I can reuse my same get_recommendations function by passing the new cosine matrix as my second argument

# ### Use this recommender to get films based from key metadata

# Users - Insert movie name into red quote marks, use full capitals and relevant spacing

# In[30]:


get_recommendations('Avatar', cosine_sim2)


# Users - Insert movie name into red quote marks, use full capitals and relevant spacing

# In[31]:


get_recommendations('The Departed', cosine_sim2)


# I have used these examples as these films both have Leonardo DiCaprio as a main actor, although the actor is the same for both films, we are presented with different recommendations based of the keywords we fed into our soup, which is why Titanic returns films like Romeo and Juliet while the Departed returns films such as shutter island.

# #  Collaborative based Filtering

# Furthered-User based filtering
# This part of the system will recommend movies to a user that similar users have liked. The result will be a prediction of the movie rating that users will give the movie based on movies they've rated before, so the system will check what movies they have rated with the fields userID, movieID, rating and timestamp. We will then enter a movie id, then the system will predict a rating.

# I will using the csv - ratings small.csv for this part

# The following cell gets all the libraries and functions that I need for my system, this system really utilises libraries rather than create a lot of calculations from scratch as there is no need to do so and it is very time consuming. The libraries below are very useful and they come with a lot of packages that really help especially with this part of the system.

# The following is original code

# In[32]:


from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
reader = Reader()
User_ratings = pd.read_csv('/Users/danielwatts/Desktop/Uni/Year 3/Computing Project/ratings_small.csv')
User_ratings.head()


# The next cell splits the data into 5 folds, this is a means of testing the data and we will see later why this is really important, the follwoing code cell splits the dataframe into k consecutive folds, each fold is then used once as a validation whilst the k-1 set fold forms the training set. This is an important part and is used very commonly in model testing. 

# In[33]:


data = Dataset.load_from_df(User_ratings[['userId', 'movieId', 'rating']], reader)
kf = KFold(n_splits=5)
kf.split(data)


# In[34]:


algo = SVD()


# This segment of the code carries of the testing of the data, once run the code evaluates the data and then generates a RMSE score, the score means how accurate my predicted rating is and for a system like mine, we are aiming for a RMSE of 0-3 because this is quite a small dataset and system. If the dataset was a lot larger then we could hope for a RMSE of 0-10 but for my system I am aiming for a RMSE of 0-3. Although RMSE could rise to a 1000.
# KEY - A good RMSE is as close to 0 as you can get.

# In[35]:


cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[36]:


svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'])


# Ulitising the surprise library, I am building a trainset which can lead me to the predictions

# In[37]:


trainset = data.build_full_trainset()
svd.fit(trainset)


# The following code prints out movies that the user in question has rated, so the user will imput a movie where the integer is, this is the movie id of the user they want to see, so I have entered id - 45.

# In[38]:


User_ratings[User_ratings['userId'] == 45]


# Users - Insert User ID into first space, then movie id, then number 3

# In[39]:


svd.predict(45, 11, 3)


# So for this example I have chosen someone with the userID of 45 and a film ID of 11, both completley random numbers, based on how other users with similar interests have rated this movie. So from the function we can see a prediction of 3.41, meanining that if user 45 watched movie 2938, there will likely be a rating of 3.41

# # Some Data Analysis and Trends

# In this part of my project I will be going into some in depth analysis of the dataset and trying to pull apart some trends that will help with my recommender system.

# I will be creating a function which will allow me to find the minimum and maximum of values that the user can input. This function will serve as the basis of this part of the project.

# In[40]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[41]:


reader = pd.read_csv('/Users/danielwatts/Desktop/Uni/Year 3/Computing Project/archive (1)/tmdb_5000_movies.csv')


# In[42]:


#Creating a new column that will calcuate the profit a film has made 
reader['Profit'] = reader['revenue'] - reader['budget']


# ### Movies with highest profability

# In[43]:


def find_minmax(x):
    min_index = reader[x].idxmin()
    high_index = reader[x].idxmax()
    high = pd.DataFrame(reader.loc[high_index,:])
    low = pd.DataFrame(reader.loc[min_index,:])
    
    #Printing the movie with the highest and lowest profits.
    print("Movie Which Has Highest "+ x + " : ",reader['original_title'][high_index])
    print("Movie Which Has Lowest "+ x + "  : ",reader['original_title'][min_index])
    return pd.concat([high,low],axis = 1)

#Calling the find_max function
find_minmax ('Profit')


# We can now see that the film with the highest budget is Avatar which is represented in the first column followed by the lowest revenue movie which is 'Wild Card'

# This graph depicts the top 10 highest revenue generating movies, this will be interesting to compare against ratings to see if we can get a correlation between revenue and ratings, this sounds obvious but its key to note that revenue is often generated by cinema sales which is before the view has seen the film for the first time. 

# In[44]:


info = pd.DataFrame(reader['revenue'].sort_values(ascending = False))
info['original_title'] = reader['original_title']
data = list(map(str,(info['original_title'])))

#extract the top 10 movies with high revenue data from the list and dataframe.
x = list(data[:10])
y = list(info['revenue'][:10])

#make the point plot and setup the title and labels.
ax = sns.pointplot(x=y,y=x)
sns.set(rc={'figure.figsize':(10,5)})
ax.set_title("Top 10 Highest Revenue Movies",fontsize = 15)
ax.set_xlabel("Revenue",fontsize = 13)
sns.set_style("darkgrid")


# In[ ]:




