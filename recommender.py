import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.width", 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


spotify_data_ = pd.read_csv("spotify_weather_data.csv")
imdb_data_ = pd.read_csv("imdb_movies_weather_data.csv")



def get_songs_by_weather(weather_col, tf_idf_col, weather_variable, num_of_recommend):
    """
        Описание:
            Цель этой функции — вычислить косинусное сходство между информацией о погоде, полученной от пользователя, и информацией о погоде в нашем наборе данных
            И рекомендовать любое количество песен из введенного набора данных в случайном порядке, в порядке убывания популярности.

        Переменные:
            dataframe: соответствующий набор данных
            weather_col: столбец погоды в наборе данных
            tf_idf_col: вычисляет частоту слов в указанном столбце в соответствующем наборе данных с помощью метода tf_idf
            weather_variable: информация о погоде, полученная от пользователя
            num_of_recommend: количество рекомендаций

    """
    dataframe = spotify_data_

    # Фильтрация собственного набора данных на основе информации о погоде, полученной от пользователя
    filtered_songs = dataframe[(dataframe[weather_col] == weather_variable)]

    # Количество запрошенных рекомендаций
    num_songs_to_recommend = num_of_recommend

    # При создании случайных индексов мы узнаем количество песен, отфильтрованных выше.
    num_songs_available = len(filtered_songs)

    # Создаём объект для преобразования текстовых данных в матрицу признаков, где каждое значение представляет собой TF-IDF (Term Frequency-Inverse Document Frequency) показатель
    # Получаем разреженную матрицу, где строки соответствуют документам, а столбцы — уникальным терминам из всего корпуса текстов. Значения в матрице представляют собой весовые коэффициенты TF-IDF для каждой пары документ-термин
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(filtered_songs[tf_idf_col])

    # Косинусное сходство измеряет угол между двумя векторами, представляющими документы. Значения сходства варьируются от -1 (полностью противоположные) до 1 (полностью идентичные)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Создание случайных индексов для выбора песни
    random_indices = np.random.choice(num_songs_available, size=num_songs_to_recommend, replace=False)

    # Получение индексов песен, отсортированных по косинусному сходству для выбранной песни
    song_indices = cosine_sim[0].argsort()

    # Выбор песен по случайным индексам
    top_songs_indices = song_indices[random_indices][::-1]

    # Получение сгенерированных выше песен из нашего набора данных
    top_songs = filtered_songs.iloc[top_songs_indices]

    # Сортировка по популярности и сброс индексов
    top_songs = top_songs.sort_values("Popularity", ascending=False).reset_index()

    # Уменьшение столбца «индекс», возникающего после сброса индекса
    del top_songs["index"]

    # Возвращаем песни
    return top_songs[
        ["Track Name", "Artist", "Album","Image", "Popularity"]
    ]


def get_movies_by_weather(weather_col, tf_idf_col, weather_variable, num_of_recommend):
    """
        Описание:
            Цель этой функции — вычислить косинусное сходство между информацией о погоде, полученной от пользователя, и информацией о погоде в нашем наборе данных
            И рекомендовать любое количество фильмов из введенного набора данных в случайном порядке, в порядке убывания популярности.
        Переменные:
            dataframe: соответствующий набор данных
            weather_col: столбец погоды в наборе данных
            tf_idf_col: вычисляет частоту слов в указанном столбце в соответствующем наборе данных с помощью метода tf_idf
            weather_variable: информация о погоде, полученная от пользователя
            num_of_recommend: количество рекомендаций
    """
    dataframe = imdb_data_
    # Фильтрация собственного набора данных на основе информации о погоде, полученной от пользователя
    filtered_movies = dataframe[(dataframe[weather_col] == weather_variable)]

    # Количество запрошенных рекомендаций
    num_movies_to_recommend = num_of_recommend

    # При создании случайных индексов мы узнаем количество фильмов, отфильтрованных выше.
    num_movies_available = len(filtered_movies)

    # Создаём объект для преобразования текстовых данных в матрицу признаков, где каждое значение представляет собой TF-IDF (Term Frequency-Inverse Document Frequency) показатель
    # Получаем разреженную матрицу, где строки соответствуют документам, а столбцы — уникальным терминам из всего корпуса текстов. Значения в матрице представляют собой весовые коэффициенты TF-IDF для каждой пары документ-термин
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(filtered_movies[tf_idf_col])

    # Косинусное сходство измеряет угол между двумя векторами, представляющими документы. Значения сходства варьируются от -1 (полностью противоположные) до 1 (полностью идентичные)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Создание случайных индексов для выбора фильмов
    random_indices = np.random.choice(num_movies_available, size=num_movies_to_recommend, replace=False)

    # Получение индексов фильмов, отсортированных по косинусному сходству для выбранного фильма
    movie_indices = cosine_sim[0].argsort()

    # Выбор фильмов по случайным индексам
    top_movies_indices = movie_indices[random_indices][::-1]

    # Получение сгенерированных выше фильмов из нашего набора данных
    top_movies = filtered_movies.iloc[top_movies_indices]

    # Сортировка по популярности и сброс индексов
    top_movies = top_movies.sort_values("Rating", ascending=False).reset_index()

    # Уменьшение столбца «индекс», возникающего после сброса индекса
    del top_movies["index"]

    # Возвращаем фильмы
    return top_movies[
        ['Title', 'Year', 'Genre', 'Description', 'Rating', 'Director', 'Votes', "Weather"]
    ]

