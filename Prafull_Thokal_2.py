

'''
QUESTIONS SEQUENCE:
    1,2,3,4,5,6,7,8,10,9(have all answers in the form of wrapper function to return the output in csv,json,png(5th) formats)
'''

import pandas as pd
global df
df = pd.read_csv("movie_meta_data.csv")
from forex_python.converter import CurrencyRates
import matplotlib.pyplot as plt

'''
1.Group movies by genres
a.Top/bottom n percentile movies according to metascore, where, ‘n’ should be a parameter passed to your function. For example, if n is 10, then you will be expected to find the movies above 90 percentile (top) and below 10 percentile (bottom) for each genre.
b.Top/bottom n percentile movies according to ‘number of imdb user votes’
'''
def get_movies_by_genre_and_rating(genres, n):
    global df
    df_clean = df.replace(-1, pd.np.nan).dropna(subset=['metascore', 'number of imdb user votes'])
    cond = df_clean['genres'].str.contains(genres)
    cond = cond.fillna(False)
    genres_data = df_clean[['title','metascore','genres']][cond] 
        
    metascore_desc = genres_data.describe(percentiles = [n/100 , (100-n)/100])
        
    bottom_n = metascore_desc['metascore']['{}%'.format(n)]
    top_n = metascore_desc['metascore']['{}%'.format(100-n)]
        
    bottom_n_movie = genres_data[genres_data['metascore'] <= bottom_n]
    top_n_movie = genres_data[genres_data['metascore'] >= top_n]

    imdbUserRating_data_desc = df_clean['number of imdb user votes'].describe(percentiles = [n/100 , (100-n)/100])

    bottom_n1 = imdbUserRating_data_desc['10%']
    top_n1 = imdbUserRating_data_desc['90%']

    bottom_n_movie_by_rating = df_clean[df_clean['number of imdb user votes']<= bottom_n1]
    top_n_movie_by_rating = df_clean[df_clean['number of imdb user votes']>= top_n1]

    return bottom_n_movie, top_n_movie, bottom_n_movie_by_rating, top_n_movie_by_rating

bottom_n_movie, top_n_movie, bottom_n_movie_by_rating, top_n_movie_by_rating = get_movies_by_genre_and_rating('Comedy', 10)
get_movies_by_genre_and_rating('Comedy', 10)
print(bottom_n_movie)
print(top_n_movie)
print(bottom_n_movie_by_rating)
print(top_n_movie_by_rating)


'''2.Movies who have won an Oscar in a particular year. For example, get the year as a parameter to your function and return all the movies that won an Oscar in that year
'''
def get_oscar_awards(year):
    global df
    regex = r"Oscar " + year
    oscar_awards = df[df['awards'].str.contains(regex, regex=True, na=False)]
    if not oscar_awards.empty:
        print("Movies that won an Oscar in " + year + ":")
        for title in oscar_awards['title']:
            print(title)
    else:
        print("No movies found that won an Oscar in " + year + ".")
get_oscar_awards('2019')

'''
3.Analyze and return n movies with highest/lowest budget
'''
def get_top_bottom_movies_by_budget(df, n):
    df = pd.read_csv("movie_meta_data.csv")
    df['budget'] = df['budget'].str.replace('$', 'USD')

    c = CurrencyRates()

    Currency_INR = c.get_rate('INR', 'USD')
    Currency_EUR = c.get_rate('EUR', 'USD')
    Currency_CAD = c.get_rate('CAD', 'USD')
    Currency_HUF = c.get_rate('HUF', 'USD')
    Currency_GBP = c.get_rate('GBP', 'USD')
    Currency_DKK = c.get_rate('DKK', 'USD')
    Currency_CNY = c.get_rate('CNY', 'USD')
    Currency_AUD = c.get_rate('AUD', 'USD')

    regex = r'^(\bUSD|\bINR|\bEUR|\bCAD|\bHUF|\bGBP|\bDKK|\bCNY|\bAUD)(\d+(,\d{3})*(\.\d+)?)\s\(estimated\)'

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

    top_n_movies = df.nlargest(n, 'budget_usd')
    bottom_n_movies = df.nsmallest(n, 'budget_usd')

    return top_n_movies[['title', 'budget_usd']], bottom_n_movies[['title', 'budget_usd']]

df = pd.read_csv("movie_meta_data.csv")
top_movies, bottom_movies = get_top_bottom_movies_by_budget(df, 10)

print(f"Top 10 movies by budget:\n{top_movies}")
print(f"Bottom 10 movies by budget:\n{bottom_movies}")


'''
4.Which countries have highest number of movies release in each year
'''
def highest_number_of_movies_by_country_per_year(df, year):
    year_data = df[df['year'] == year]
    country_counts = year_data['countries'].str.split(',').explode().str.strip().value_counts()
    top_country = country_counts.index[0]
    results = {}
    for country, count in country_counts.items():
        results[country] = count
    results['Top Country'] = top_country
    return results

df = pd.read_csv("movie_meta_data.csv")
highest_number_of_movies_by_country_per_year(df, 2010)


'''
5.Analyze if there is any relationship between the imdb user rating and number of awards received'''
global df
df['awards_count'] = df['awards'].str.count(',') + 1
df_filtered = df[df['imdb user rating'] != -1]
plt.scatter(df_filtered['imdb user rating'], df_filtered['awards_count'])
plt.xlabel('IMDB User Rating')
plt.ylabel('Awards Count')
plt.title('Relationship between IMDB User Rating and Awards Count')
plt.show()

'''
6.Return akas of a specified movie in a specified region
'''
def get_movie_aka_name(movie: str, region: str) -> str:
    region_code_start = '('
    region_code_end = ')'
    movie_row = df[df['title'] == movie]
    akas_list = movie_row['akas'].values[0].split(',')
    for aka_name in akas_list:
        region_code = aka_name[aka_name.find(region_code_start) + 1: aka_name.find(region_code_end)]
        if region_code == region:
            aka_name_without_region_code = aka_name[: aka_name.find(region_code_start)].strip()
            return aka_name_without_region_code
    return "No aka name found for the specified region."

aka_name = get_movie_aka_name('A Night at the Roxbury', 'Uruguay')
print(aka_name)

'''
7.Movies released on, before or after a given year (take year as a parameter)
'''
def movies_by_release_year(df, year):
    df = pd.read_csv("movie_meta_data.csv")
    on_year = list(df[df['year'] == year]['title'])
    before_year = list(df[df['year'] < year]['title'])
    after_year = list(df[df['year'] > year]['title'])
    return on_year, before_year, after_year

on_year, before_year, after_year = movies_by_release_year(df, 2000)
print(on_year, "\n\n\n\n\n")
print(before_year, "\n\n\n\n\n")
print(after_year, "\n\n\n\n\n")

'''
8.Which director has made directed most number of oscar winning movies
'''
def get_most_oscar_director():
    global df
    min_year = 1913
    max_year = 2023
    oscar_winners = pd.DataFrame(columns=df.columns)
    for year in range(min_year, max_year):
        regex = r"Oscar " + str(year)
        oscar_awards = df[df['awards'].str.contains(regex, regex=True, na=False)]
        if not oscar_awards.empty:
            oscar_winners = oscar_winners.append(oscar_awards)
    director_movies = {}
    for index, row in oscar_winners.iterrows():
        directors = row['directors'].split(",")
        for director in directors:
            if director.strip() in director_movies:
                director_movies[director.strip()].append(row['title'])
            else:
                director_movies[director.strip()] = [row['title']]

    most_oscar_director = max(director_movies, key=lambda x: len(director_movies[x]))
    most_oscar_count = len(director_movies[most_oscar_director])

    result = "The director who has directed the most number of Oscar-winning movies is " + most_oscar_director + ".\n"
    result += "Number of Oscar-winning movies directed by " + most_oscar_director + ": " + str(most_oscar_count)
    return result

get_most_oscar_director()


''''
10.For each genre of movies identify:
a.Average budget: addition of all budgets of each movie / no of movies in that genre
'''
def get_genre_budgets(df):
    df['budget'] = df['budget'].str.replace('$', 'USD')

    c = CurrencyRates()
    Currency_INR = c.get_rate('INR', 'USD')
    Currency_EUR = c.get_rate('EUR', 'USD')
    Currency_CAD = c.get_rate('CAD', 'USD')
    Currency_HUF = c.get_rate('HUF', 'USD')
    Currency_GBP = c.get_rate('GBP', 'USD')
    Currency_DKK = c.get_rate('DKK', 'USD')
    Currency_CNY = c.get_rate('CNY', 'USD')
    Currency_AUD = c.get_rate('AUD', 'USD')

    regex = r'^(\bUSD|\bINR|\bEUR|\bCAD|\bHUF|\bGBP|\bDKK|\bCNY|\bAUD)(\d+(,\d{3})*(\.\d+)?)\s\(estimated\)'

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

    df["genres"] = df["genres"].str.split(", ")
    df = df.explode("genres")
    genre_budgets = df.groupby("genres")["budget_usd"].mean()
    return genre_budgets

get_genre_budgets(df)


'''
b.Most common director: for one particular genre who had directed most movies in that genre
c.Most common cast member (actor): for one particular genre who had been part of most movies in that genre
'''
def get_most_common_director_or_actor(genre):
    df = pd.read_csv('movie_meta_data.csv')
    df['genres'] = df['genres'].fillna('Unknown')
    genre_df = df[df['genres'].str.contains(genre)]
    director_counts = genre_df['directors'].str.split(',').explode().value_counts()
    most_common_director = director_counts.index[0]
    df['cast'] = df['cast'].fillna('')
    genre_df = df[df['genres'].str.contains(genre)]
    cast_count = {}
    for cast_list in genre_df['cast']:
        for cast_member in cast_list.split(','):
            cast_member = cast_member.strip()
            if cast_member in cast_count:
                cast_count[cast_member] += 1
            else:
                cast_count[cast_member] = 1
    most_common_cast_member = max(cast_count, key=cast_count.get)

    print(f"The most common director in the {genre} genre is {most_common_director}.")
    print(f"The most common cast member for {genre} genre is {most_common_cast_member} with a count of {cast_count[most_common_cast_member]}")

get_most_common_director_or_actor('Fantasy')


'''
9.Write a wrapper function to return the output in following formats
a.CSV, Json (all questions except question 5)
b.png(for question 5)
'''
#1)
import pandas as pd
import json
import csv
import os
from forex_python.converter import CurrencyRates
import matplotlib.pyplot as plt

def highest_number_of_movies_by_country_per_year(df, year, output_format='csv', output_file=None):
    year_data = df[df['year'] == year]
    country_counts = year_data['countries'].str.split(',').explode().str.strip().value_counts()
    top_country = country_counts.index[0]
    results = {}
    for country, count in country_counts.items():
        results[country] = count
    results['Top Country'] = top_country
    if output_format == 'csv':
        if output_file is not None:
            results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Count'])
            results_df.to_csv(output_file)
            print(f"Results saved to {output_file}")
        else:
            print(results)
    elif output_format == 'json':
        if output_file is not None:
            with open(output_file, 'w') as f:
                json.dump(results, f)
            print(f"Results saved to {output_file}")
        else:
            print(json.dumps(results, indent=4))
    else:
        print("Invalid output format. Please choose 'csv' or 'json'.")

#2)
def get_oscar_awards(year, output_format='csv', output_file=None):
    df = pd.read_csv("movie_meta_data.csv")
    regex = r"Oscar " + year
    oscar_awards = df[df['awards'].str.contains(regex, regex=True, na=False)]
    if not oscar_awards.empty:
        if output_file is None:
            output_file = f"oscar_awards_{year}.{output_format}"
        try:
            if output_format == 'csv':
                oscar_awards.to_csv(output_file, index=False)
            elif output_format == 'json':
                oscar_awards.to_json(output_file, orient='records')
            else:
                print("Invalid output format. Please specify either 'csv' or 'json'.")
        except OSError:
            print(f"Error: could not write output to {output_file}.")
    else:
        print("No movies found that won an Oscar in " + year + ".")

#3)
def get_top_bottom_movies_by_budget(df, n):
    df['budget'] = df['budget'].str.replace('$', 'USD')

    c = CurrencyRates()

    Currency_INR = c.get_rate('INR', 'USD')
    Currency_EUR = c.get_rate('EUR', 'USD')
    Currency_CAD = c.get_rate('CAD', 'USD')
    Currency_HUF = c.get_rate('HUF', 'USD')
    Currency_GBP = c.get_rate('GBP', 'USD')
    Currency_DKK = c.get_rate('DKK', 'USD')
    Currency_CNY = c.get_rate('CNY', 'USD')
    Currency_AUD = c.get_rate('AUD', 'USD')

    regex = r'^(\bUSD|\bINR|\bEUR|\bCAD|\bHUF|\bGBP|\bDKK|\bCNY|\bAUD)(\d+(,\d{3})*(\.\d+)?)\s\(estimated\)'

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
                                    r'\bUSD\b': 1,
                                    r'\bINR\b': Currency_INR,
                                    r'\bEUR\b': Currency_EUR,
                                    r'\bCAD\b': Currency_CAD,
                                    r'\bHUF\b': Currency_HUF,
                                    r'\bGBP\b': Currency_GBP,
                                    r'\bDKK\b': Currency_DKK,
                                    r'\bCNY\b': Currency_CNY,
                                    r'\bAUD\b': Currency_AUD
                                }))

    top_n_movies = df.nlargest(n, 'budget_usd')
    bottom_n_movies = df.nsmallest(n, 'budget_usd')

    return top_n_movies[['title', 'budget_usd']], bottom_n_movies[['title', 'budget_usd']]

def write_output_to_file(output, file_format, file_name):
    if file_format == 'csv':
        output.to_csv(file_name, index=False)
    elif file_format == 'json':
        output.to_json(file_name, orient='records')

#4)
def highest_number_of_movies_by_country_per_year(df, year, output_format='dict'):
    year_data = df[df['year'] == year]
    country_counts = year_data['countries'].str.split(',').explode().str.strip().value_counts()
    top_country = country_counts.index[0]
    results = {}
    for country, count in country_counts.items():
        results[country] = count
    results['Top Country'] = top_country
    if output_format == 'dict':
        return results
    elif output_format == 'csv':
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Count'])
        return results_df.to_csv(index_label='Country')
    elif output_format == 'json':
        return json.dumps(results, indent=4)
    else:
        raise ValueError('Invalid output format. Choose from "dict", "csv", or "json".')
def write_output_to_file(output, output_format, file_path):
    with open(file_path, 'w') as f:
        if output_format == 'csv':
            f.write(output)
        elif output_format == 'json':
            json.dump(output, f, indent=4)
        else:
            raise ValueError('Invalid output format. Choose from "csv" or "json".')

#5)
def save_imdb_awards_plot(filename):
    df = pd.read_csv("movie_meta_data.csv")
    df['awards_count'] = df['awards'].str.count(',') + 1
    df_filtered = df[df['imdb user rating'] != -1]
    plt.scatter(df_filtered['imdb user rating'], df_filtered['awards_count'])
    plt.xlabel('IMDB User Rating')
    plt.ylabel('Awards Count')
    plt.title('Relationship between IMDB User Rating and Awards Count')
    plt.savefig(filename)
    plt.show()
    
#6)
def get_movie_aka_name(movie: str, region: str, output_format: str = 'csv') -> dict:
    folder_path = r'C:\Users\Prafull_Thokal'
    df = pd.read_csv(f"{folder_path}/movie_meta_data.csv")
    
    region_code_start = '('
    region_code_end = ')'
    movie_row = df[df['title'] == movie]
    akas_list = movie_row['akas'].values[0].split(',')
    for aka_name in akas_list:
        region_code = aka_name[aka_name.find(region_code_start) + 1: aka_name.find(region_code_end)]
        if region_code == region:
            aka_name_without_region_code = aka_name[: aka_name.find(region_code_start)].strip()

            if output_format == 'csv':
                return {'movie': movie, 'aka_name': aka_name_without_region_code, 'region': region}
            elif output_format == 'json':
                return {'movie': movie, 'aka_name': aka_name_without_region_code, 'region': region}

    return {'error': "No aka name found for the specified region."}

#7)
def movies_by_release_year(df, year, output_format=None, filename=None):
    on_year_df = df[df['year'] == year][['title', 'year']]
    before_year_df = df[df['year'] < year][['title', 'year']]
    after_year_df = df[df['year'] > year][['title', 'year']]
    if output_format == 'csv':
        on_year_df.to_csv(f'{filename}_on_year.csv', index=False)
        before_year_df.to_csv(f'{filename}_before_year.csv', index=False)
        after_year_df.to_csv(f'{filename}_after_year.csv', index=False)
    elif output_format == 'json':
        on_year_data = {'on_year': on_year_df.to_dict(orient='records')}
        with open(f'{filename}_on_year.json', 'w') as f:
            json.dump(on_year_data, f)  
        before_year_data = {'before_year': before_year_df.to_dict(orient='records')}
        with open(f'{filename}_before_year.json', 'w') as f:
            json.dump(before_year_data, f)   
        after_year_data = {'after_year': after_year_df.to_dict(orient='records')}
        with open(f'{filename}_after_year.json', 'w') as f:
            json.dump(after_year_data, f)
    else:
        return on_year_df, before_year_df, after_year_df

#8)
def get_most_oscar_director(output_format=None, filename=None):
    df = pd.read_csv("movie_meta_data.csv")
    min_year = 1913
    max_year = 2023
    oscar_winners = pd.DataFrame(columns=df.columns)
    for year in range(min_year, max_year):
        regex = r"Oscar " + str(year)
        oscar_awards = df[df['awards'].str.contains(regex, regex=True, na=False)]
        if not oscar_awards.empty:
            oscar_winners = pd.concat([oscar_winners, oscar_awards], ignore_index=True)
    director_movies = {}
    for index, row in oscar_winners.iterrows():
        directors = row['directors'].split(",")
        for director in directors:
            if director.strip() in director_movies:
                director_movies[director.strip()].append(row['title'])
            else:
                director_movies[director.strip()] = [row['title']]

    most_oscar_director = max(director_movies, key=lambda x: len(director_movies[x]))
    most_oscar_count = len(director_movies[most_oscar_director])

    result = "The director who has directed the most number of Oscar-winning movies is " + most_oscar_director + ".\n"
    result += "Number of Oscar-winning movies directed by " + most_oscar_director + ": " + str(most_oscar_count)
    if output_format == 'csv':
        try:
            with open(filename, 'w') as f:
                pass
        except:
            print("Error: Invalid file path.")
            return None
        df_csv = pd.DataFrame({'Director': [most_oscar_director], 'Number of Oscar-Winning Movies': [most_oscar_count]})
        df_csv.to_csv(filename, index=False)
        return None
    elif output_format == 'json':
        try:
            with open(filename, 'w') as f:
                pass
        except:
            print("Error: Invalid file path.")
            return None
        data = {'Director': most_oscar_director, 'Number of Oscar-Winning Movies': most_oscar_count}
        with open(filename, 'w') as f:
            json.dump(data, f)
        return None
    else:
        return result


#10)a)
def get_genre_budgets(df):
    df['budget'] = df['budget'].str.replace('$', 'USD')
    
    c = CurrencyRates()
    
    Currency_INR = c.get_rate('INR', 'USD')
    Currency_EUR = c.get_rate('EUR', 'USD')
    Currency_CAD = c.get_rate('CAD', 'USD')
    Currency_HUF = c.get_rate('HUF', 'USD')
    Currency_GBP = c.get_rate('GBP', 'USD')
    Currency_DKK = c.get_rate('DKK', 'USD')
    Currency_CNY = c.get_rate('CNY', 'USD')
    Currency_AUD = c.get_rate('AUD', 'USD')

    regex = r'^(\bUSD|\bINR|\bEUR|\bCAD|\bHUF|\bGBP|\bDKK|\bCNY|\bAUD)(\d+(,\d{3})*(\.\d+)?)\s\(estimated\)'

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
                                        r'\bUSD\b': 1,
                                        r'\bINR\b': Currency_INR,
                                        r'\bEUR\b': Currency_EUR,
                                        r'\bCAD\b': Currency_CAD,
                                        r'\bHUF\b': Currency_HUF,
                                        r'\bGBP\b': Currency_GBP,
                                        r'\bDKK\b': Currency_DKK,
                                        r'\bCNY\b': Currency_CNY,
                                        r'\bAUD\b': Currency_AUD
                                    }))

    df["genres"] = df["genres"].str.split(", ")
    df = df.explode("genres")
    genre_budgets = df.groupby("genres")["budget_usd"].mean()
    genre_budgets.to_csv('genre_budgets.csv')
    genre_budgets.to_json('genre_budgets.json')
    return genre_budgets

#10)b,c)
def get_most_common_director_or_actor(genre, output_format='csv'):
    df = pd.read_csv('movie_meta_data.csv')
    df['genres'] = df['genres'].fillna('Unknown')
    genre_df = df[df['genres'].str.contains(genre)]
    director_counts = genre_df['directors'].str.split(',').explode().value_counts()
    most_common_director = director_counts.index[0]
    df['cast'] = df['cast'].fillna('')
    genre_df = df[df['genres'].str.contains(genre)]
    cast_count = {}
    for cast_list in genre_df['cast']:
        for cast_member in cast_list.split(','):
            cast_member = cast_member.strip()
            if cast_member in cast_count:
                cast_count[cast_member] += 1
            else:
                cast_count[cast_member] = 1

    most_common_cast_member = max(cast_count, key=cast_count.get)

    results = {
        'most_common_director': most_common_director,
        'most_common_cast_member': most_common_cast_member,
        'cast_member_count': cast_count[most_common_cast_member]
    }

    if output_format == 'csv':
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['value'])
        results_df.to_csv('most_common_director_or_actor.csv')
    elif output_format == 'json':
        with open('most_common_director_or_actor.json', 'w') as f:
            json.dump(results, f)
    return results

def write_output_to_file(results, output_format, file_path):
    if output_format == 'csv':
        results_df = pd.DataFrame(list(results.items()), columns=['key', 'value'])
        results_df.to_csv(file_path, index=False)
    elif output_format == 'json':
        with open(file_path, 'w') as f:
            json.dump(results, f)
            


def main():
    folder_path = r'C:\Users\Prafull_Thokal'
    df = pd.read_csv(os.path.join(folder_path, 'movie_meta_data.csv'))

    # 1) Highest number of movies by country per year
    year = 2010
    output_file_csv = os.path.join(folder_path, f'highest_number_of_movies_by_country_{year}.csv')
    highest_number_of_movies_by_country_per_year(df, year, output_format='csv', output_file=output_file_csv)
    output_file_json = os.path.join(folder_path, f'highest_number_of_movies_by_country_{year}.json')
    highest_number_of_movies_by_country_per_year(df, year, output_format='json', output_file=output_file_json)

    # 2) Oscar awards
    year = '2019'
    output_file_csv = os.path.join(folder_path, f'oscar_awards_{year}.csv')
    get_oscar_awards(year, output_format='csv', output_file=output_file_csv)
    output_file_json = os.path.join(folder_path, f'oscar_awards_{year}.json')
    get_oscar_awards(year, output_format='json', output_file=output_file_json)

    # 3) Top and bottom movies by budget
    top_movies, bottom_movies = get_top_bottom_movies_by_budget(df, 10)
    write_output_to_file(top_movies, 'csv', os.path.join(folder_path, 'top_movies.csv'))
    write_output_to_file(bottom_movies, 'csv', os.path.join(folder_path, 'bottom_movies.csv'))
    write_output_to_file(top_movies, 'json', os.path.join(folder_path, 'top_movies.json'))
    write_output_to_file(bottom_movies, 'json', os.path.join(folder_path, 'bottom_movies.json'))

    # 4) Highest number of movies by country per year (printing to console)
    year = 2010
    results = highest_number_of_movies_by_country_per_year(df, year, output_format='csv')
    write_output_to_file(results, 'csv', os.path.join(folder_path, f'highest_number_of_movies_by_country_{year}_console.csv'))
    results = highest_number_of_movies_by_country_per_year(df, year, output_format='json')
    write_output_to_file(results, 'json', os.path.join(folder_path, f'highest_number_of_movies_by_country_{year}_console.json'))

    # 5) IMDB awards plot
    save_imdb_awards_plot(os.path.join(folder_path, 'my_plot.png'))

    # 6) Get movie AKA name
    aka_name_csv = get_movie_aka_name('A Night at the Roxbury', 'France', 'csv')
    write_output_to_file(aka_name_csv, 'csv', os.path.join(folder_path, 'aka_name.csv'))
    aka_name_json = get_movie_aka_name('A Night at the Roxbury', 'France', 'json')
    write_output_to_file(aka_name_json, 'json', os.path.join(folder_path, 'aka_name.json'))

    # 7) Movies by release year
    year = 2000
    movies_by_release_year(df, year, output_format='csv', filename=os.path.join(folder_path, f'movies_by_release_year_{year}.csv'))
    movies_by_release_year(df, year, output_format='json', filename=os.path.join(folder_path, f'movies_by_release_year_{year}.json'))

    #8)
    csv_file_path = os.path.join(folder_path, 'most_oscar_directed_movies.csv')
    json_file_path = os.path.join(folder_path, 'most_oscar_directed_movies.json')
    output_csv = get_most_oscar_director(output_format='csv', filename=csv_file_path)
    if output_csv:
        write_output_to_file(output_csv, 'csv', csv_file_path)
    output_json = get_most_oscar_director(output_format='json', filename=json_file_path)
    if output_json:
        write_output_to_file(output_json, 'json', json_file_path)
        
    #10)a)
    write_output_to_file(get_genre_budgets(df), 'csv', os.path.join(folder_path, 'genre_budgets.csv'))

    #10)b,c)
    results_csv = get_most_common_director_or_actor('Fantasy')
    write_output_to_file(results_csv, 'csv', os.path.join(folder_path, 'most_common_director_or_actor_fantasy.csv'))
    results_json = get_most_common_director_or_actor('Fantasy')
    write_output_to_file(results_json, 'json', os.path.join(folder_path, 'most_common_director_or_actor_fantasy.json'))


if __name__ == '__main__':
    main()
