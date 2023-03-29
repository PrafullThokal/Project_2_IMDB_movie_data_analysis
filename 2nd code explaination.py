# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:41:58 2023

@author: Prafull_Thokal
"""

1st explaination:
def get_movies_by_genre_and_rating(genres, n):
    global df # Accessing the global variable 'df' defined outside the function
    df_clean = df.replace(-1, pd.np.nan).dropna(subset=['metascore', 'number of imdb user votes']) # Clean the data by replacing -1 with NaN and dropping rows with missing values in 'metascore' and 'number of imdb user votes' columns
    cond = df_clean['genres'].str.contains(genres) # Create a boolean mask of rows where the 'genres' column contains the specified 'genres'
    cond = cond.fillna(False) # Replace any missing values in the boolean mask with False
    genres_data = df_clean[['title','metascore','genres']][cond] # Extract the 'title', 'metascore', and 'genres' columns for rows where the 'genres' column contains the specified 'genres'
        
    metascore_desc = genres_data.describe(percentiles = [n/100 , (100-n)/100]) # Compute the descriptive statistics of 'metascore' column for the extracted rows, and include the specified percentile values
        
    bottom_n = metascore_desc['metascore']['{}%'.format(n)] # Retrieve the 'metascore' value at the specified percentile (n%)
    top_n = metascore_desc['metascore']['{}%'.format(100-n)] # Retrieve the 'metascore' value at the specified percentile (100-n%)
        
    bottom_n_movie = genres_data[genres_data['metascore'] <= bottom_n] # Create a new DataFrame of movies with 'metascore' values less than or equal to the 'bottom_n' value
    top_n_movie = genres_data[genres_data['metascore'] >= top_n] # Create a new DataFrame of movies with 'metascore' values greater than or equal to the 'top_n' value

    imdbUserRating_data_desc = df_clean['number of imdb user votes'].describe(percentiles = [n/100 , (100-n)/100]) # Compute the descriptive statistics of 'number of imdb user votes' column for all rows in the cleaned DataFrame, and include the specified percentile values

    bottom_n1 = imdbUserRating_data_desc['10%'] # Retrieve the 'number of imdb user votes' value at the 10th percentile
    top_n1 = imdbUserRating_data_desc['90%'] # Retrieve the 'number of imdb user votes' value at the 90th percentile

    bottom_n_movie_by_rating = df_clean[df_clean['number of imdb user votes']<= bottom_n1] # Create a new DataFrame of movies with 'number of imdb user votes' values less than or equal to the 10th percentile
    top_n_movie_by_rating = df_clean[df_clean['number of imdb user votes']>= top_n1] # Create a new DataFrame of movies with 'number of imdb user votes' values greater than or equal to the 90th percentile

    return bottom_n_movie, top_n_movie, bottom_n_movie_by_rating, top_n_movie_by_rating # Return the four DataFrames created above

2nd  explaination:
    # Construct a regular expression to match the year in the awards column
regex = r"Oscar " + year

# Use the regular expression to filter the rows of 'df' that contain the year in the awards column
oscar_awards = df[df['awards'].str.contains(regex, regex=True, na=False)]

# If there are movies that won an Oscar in the given year, print their titles
if not oscar_awards.empty:
    print("Movies that won an Oscar in " + year + ":")
    for title in oscar_awards['title']:
        print(title)
# If there are no movies that won an Oscar in the given year, print a message indicating this
else:
    print("No movies found that won an Oscar in " + year + ".")


3rd:
    
# Replacing the '$' sign with 'USD' in the budget column
df['budget'] = df['budget'].str.replace('$', 'USD')

# Initializing the 'CurrencyRates' object
c = CurrencyRates()

# Getting the currency exchange rate of various currencies with respect to USD
Currency_INR = c.get_rate('INR', 'USD')
Currency_EUR = c.get_rate('EUR', 'USD')
Currency_CAD = c.get_rate('CAD', 'USD')
Currency_HUF = c.get_rate('HUF', 'USD')
Currency_GBP = c.get_rate('GBP', 'USD')
Currency_DKK = c.get_rate('DKK', 'USD')
Currency_CNY = c.get_rate('CNY', 'USD')
Currency_AUD = c.get_rate('AUD', 'USD')

# Regular expression to extract currency symbol and budget value
regex = r'^(\bUSD|\bINR|\bEUR|\bCAD|\bHUF|\bGBP|\bDKK|\bCNY|\bAUD)(\d+(,\d{3})*(\.\d+)?)\s\(estimated\)'

# Converting budget values to USD
df['budget_usd'] = df['budget'].str.extract(regex, expand=False)[1] \
                        .str.replace(',', '').astype(float) \
                        .replace({
                            'USD': 1,
                            'INR': Currency_INR,
                            'EUR': Currency_EUR,
                            'CAD': Currency_CAD,
                            'HUF': Currency_HUF,
                            'GBP': Currency_GBP,
                            'DKK': Currency_DKK,
                            'CNY': Currency_CNY,
                            'AUD': Currency_AUD
                        }).mul(df['budget'].str.extract(regex, expand=False)[0] \
                            .replace({
                                'USD': 1,
                                'INR': Currency_INR,
                                'EUR': Currency_EUR,
                                'CAD': Currency_CAD,
                                'HUF': Currency_HUF,
                                'GBP': Currency_GBP,
                                'DKK': Currency_DKK,
                                'CNY': Currency_CNY,
                                'AUD': Currency_AUD
                            }).replace({
                                '\bUSD\b': 1,
                                '\bINR\b': Currency_INR,
                                '\bEUR\b': Currency_EUR,
                                '\bCAD\b': Currency_CAD,
                                '\bHUF\b': Currency_HUF,
                                '\bGBP\b': Currency_GBP,
                                '\bDKK\b': Currency_DKK,
                                '\bCNY\b': Currency_CNY,
                                '\bAUD\b': Currency_AUD
                            }))

# Getting the top n movies with the highest budget value
top_n_movies = df.nlargest(n, 'budget_usd')

# Getting the bottom n movies with the lowest budget value
bottom_n_movies = df.nsmallest(n, 'budget_usd')

# Returning the top and bottom n movies with their respective budget values
return top_n_movies[['title', 'budget_usd']], bottom_n_movies[['title', 'budget_usd']]


4th:
def highest_number_of_movies_by_country_per_year(df, year):
    # Filter the data for the given year
    year_data = df[df['year'] == year]
    
    # Split the countries column by comma, explode the resulting list, remove whitespace, and count the occurrence of each country
    country_counts = year_data['countries'].str.split(',').explode().str.strip().value_counts()
    
    # Get the country with the highest count
    top_country = country_counts.index[0]
    
    # Create a dictionary to store the counts for each country and the top country
    results = {}
    for country, count in country_counts.items():
        results[country] = count
    results['Top Country'] = top_country
    
    # Return the dictionary
    return results

# Read the movie meta data from a CSV file and call the function to get the highest number of movies by country for the year 2010
df = pd.read_csv("movie_meta_data.csv")
highest_number_of_movies_by_country_per_year(df, 2010)


5th:
# Set the global variable 'df' to the pandas DataFrame containing the movie data
global df

# Add a new column 'awards_count' to the DataFrame that counts the number of awards for each movie
df['awards_count'] = df['awards'].str.count(',') + 1

# Filter the DataFrame to exclude movies with an IMDB user rating of -1 (which indicates no rating)
df_filtered = df[df['imdb user rating'] != -1]

# Create a scatter plot with the IMDB user rating on the x-axis and the awards count on the y-axis
plt.scatter(df_filtered['imdb user rating'], df_filtered['awards_count'])

# Label the x-axis, y-axis, and title of the plot
plt.xlabel('IMDB User Rating')
plt.ylabel('Awards Count')
plt.title('Relationship between IMDB User Rating and Awards Count')

# Display the plot
plt.show()

6th:
def get_movie_aka_name(movie: str, region: str) -> str:
    # Define the start and end characters for the region code in the aka name
    region_code_start = '('
    region_code_end = ')'
    
    # Filter the DataFrame to get the row for the specified movie
    movie_row = df[df['title'] == movie]
    
    # Split the aka names by comma and loop through them to find the one that matches the specified region code
    akas_list = movie_row['akas'].values[0].split(',')
    for aka_name in akas_list:
        region_code = aka_name[aka_name.find(region_code_start) + 1: aka_name.find(region_code_end)]
        if region_code == region:
            aka_name_without_region_code = aka_name[: aka_name.find(region_code_start)].strip()
            return aka_name_without_region_code
    
    # Return a message if no aka name is found for the specified region
    return "No aka name found for the specified region."

# Call the function to get the aka name for the specified movie and region
aka_name = get_movie_aka_name('A Night at the Roxbury', 'Uruguay')
print(aka_name)

7th:
def movies_by_release_year(df, year):
    # Load the movie_meta_data.csv file into a DataFrame
    df = pd.read_csv("movie_meta_data.csv")
    
    # Get lists of movie titles released on, before, and after the specified year
    on_year = list(df[df['year'] == year]['title'])
    before_year = list(df[df['year'] < year]['title'])
    after_year = list(df[df['year'] > year]['title'])
    
    # Return the lists
    return on_year, before_year, after_year

# Call the function to get lists of movies released on, before, and after the year 2000
on_year, before_year, after_year = movies_by_release_year(df, 2000)

# Print the lists
print(on_year, "\n\n\n\n\n")
print(before_year, "\n\n\n\n\n")
print(after_year, "\n\n\n\n\n")

8th:
The function get_most_oscar_director aims to find the director who has 
directed the most number of Oscar-winning movies in the given dataset. 
Here are the comments for the code:

def get_most_oscar_director():
    global df
    min_year = 1913
    max_year = 2023
The function starts by setting the minimum and maximum year to be considered for the search.

    oscar_winners = pd.DataFrame(columns=df.columns)
An empty Pandas DataFrame is created to hold the data of Oscar-winning movies.

    for year in range(min_year, max_year):
        regex = r"Oscar " + str(year)
        oscar_awards = df[df['awards'].str.contains(regex, regex=True, na=False)]
        if not oscar_awards.empty:
            oscar_winners = oscar_winners.append(oscar_awards)
A loop iterates over all the years between the minimum and maximum year set above.
 In each iteration, the code searches for the rows containing the substring "Oscar" followed 
 by the year in the 'awards' column using regular expressions. If any such row is found,
 it is appended to the oscar_winners DataFrame.

    director_movies = {}
    for index, row in oscar_winners.iterrows():
        directors = row['directors'].split(",")
        for director in directors:
            if director.strip() in director_movies:
                director_movies[director.strip()].append(row['title'])
            else:
                director_movies[director.strip()] = [row['title']]
The code then creates a dictionary director_movies to hold the directors and their corresponding movies. It iterates over each row of oscar_winners DataFrame, splits the 'directors' column by ',' to get all the directors for that movie. It then adds the movie to the list of movies corresponding to each director in the director_movies dictionary.


    most_oscar_director = max(director_movies, key=lambda x: len(director_movies[x]))
    most_oscar_count = len(director_movies[most_oscar_director])

    result = "The director who has directed the most number of Oscar-winning movies is " + most_oscar_director + ".\n"
    result += "Number of Oscar-winning movies directed by " + most_oscar_director + ": " + str(most_oscar_count)
    return result

Finally, the code finds the director who has directed the most 
number of Oscar-winning movies by getting the key with the maximum 
value of the director_movies dictionary using the max() function with a 
lambda function as the key argument. It then stores the number of Oscar-winning 
movies directed by the director in the variable most_oscar_count. The function
 returns a string with the name of the director and the number of Oscar-winning 
 movies they have directed.
 
10th:
a)
from forex_python.converter import CurrencyRates
import pandas as pd

def get_genre_budgets(df):
    # Replace dollar sign in budget column
    df['budget'] = df['budget'].str.replace('$', 'USD')

    # Get exchange rates for various currencies
    c = CurrencyRates()
    Currency_INR = c.get_rate('INR', 'USD')
    Currency_EUR = c.get_rate('EUR', 'USD')
    Currency_CAD = c.get_rate('CAD', 'USD')
    Currency_HUF = c.get_rate('HUF', 'USD')
    Currency_GBP = c.get_rate('GBP', 'USD')
    Currency_DKK = c.get_rate('DKK', 'USD')
    Currency_CNY = c.get_rate('CNY', 'USD')
    Currency_AUD = c.get_rate('AUD', 'USD')

    # Define regex pattern to extract currency code and amount from budget column
    regex = r'^(\bUSD|\bINR|\bEUR|\bCAD|\bHUF|\bGBP|\bDKK|\bCNY|\bAUD)(\d+(,\d{3})*(\.\d+)?)\s\(estimated\)'

    # Extract currency code and amount from budget column using regex
    # Convert amount to float and convert to USD using exchange rates
    df['budget_usd'] = df['budget'].str.extract(regex, expand=False)[1] \
                                .str.replace(',', '').astype(float) \
                                .replace({
                                    'USD': 1,
                                    'INR': Currency_INR,
                                    'EUR': Currency_EUR,
                                    'CAD': Currency_CAD,
                                    'HUF': Currency_HUF,
                                    'GBP': Currency_GBP,
                                    'DKK': Currency_DKK,
                                    'CNY': Currency_CNY,
                                    'AUD': Currency_AUD
                                }).mul(df['budget'].str.extract(regex, expand=False)[0] \
                                    .replace({
                                        'USD': 1,
                                        'INR': Currency_INR,
                                        'EUR': Currency_EUR,
                                        'CAD': Currency_CAD,
                                        'HUF': Currency_HUF,
                                        'GBP': Currency_GBP,
                                        'DKK': Currency_DKK,
                                        'CNY': Currency_CNY,
                                        'AUD': Currency_AUD
                                    }).replace({
                                        '\bUSD\b': 1,
                                        '\bINR\b': Currency_INR,
                                        '\bEUR\b': Currency_EUR,
                                        '\bCAD\b': Currency_CAD,
                                        '\bHUF\b': Currency_HUF,
                                        '\bGBP\b': Currency_GBP,
                                        '\bDKK\b': Currency_DKK,
                                        '\bCNY\b': Currency_CNY,
                                        '\bAUD\b': Currency_AUD
                                    }))

    # Split genres column by comma and create new row for each genre
    df["genres"] = df["genres"].str.split(", ")
    df = df.explode("genres")

    # Group by genre and calculate mean budget in USD
    genre_budgets = df.groupby("genres")["budget_usd"].mean()
    return genre_budgets

# Call function with movie data
get_genre_budgets(df)

b,c)
import pandas as pd
import numpy as np

def get_most_common_director_or_actor(genre):
# load the movie meta data csv file
df = pd.read_csv('movie_meta_data.csv')

python
Copy code
# replace missing genre data with 'Unknown'
df['genres'] = df['genres'].fillna('Unknown')

# select only the rows that contain the target genre
genre_df = df[df['genres'].str.contains(genre)]

# count the number of movies each director has directed for the target genre
director_counts = genre_df['directors'].str.split(',').explode().value_counts()

# get the director with the highest count
most_common_director = director_counts.index[0]

# fill any missing cast data with an empty string
df['cast'] = df['cast'].fillna('')

# select only the rows that contain the target genre
genre_df = df[df['genres'].str.contains(genre)]

# count the number of movies each cast member has appeared in for the target genre
cast_count = {}
for cast_list in genre_df['cast']:
    for cast_member in cast_list.split(','):
        cast_member = cast_member.strip()
        if cast_member in cast_count:
            cast_count[cast_member] += 1
        else:
            cast_count[cast_member] = 1

# get the cast member with the highest count
most_common_cast_member = max(cast_count, key=cast_count.get)

# print the results
print(f"The most common director in the {genre} genre is {most_common_director}.")
print(f"The most common cast member for {genre} genre is {most_common_cast_member} with a count of {cast_count[most_common_cast_member]}")