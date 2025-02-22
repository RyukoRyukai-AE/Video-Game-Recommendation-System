from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from rapidfuzz import process
from spellchecker import SpellChecker

class Recommender:
    def __init__(self, vectorizer: TfidfVectorizer, game_names: list[str], nn_model: NearestNeighbors, scaled_features: pd.DataFrame, df: pd.DataFrame):
        self.__vectorizer = vectorizer
        self.__game_names = game_names
        self.__df = df
        self.__vg_distances, self.__vg_indices = nn_model.kneighbors(scaled_features)
        self.__game_title_vectors = vectorizer.transform(game_names)

    def VideoGameTitleRecommender(self, video_game_name: str):
        query_vector = self.__vectorizer.transform([video_game_name])
        similarity_scores = cosine_similarity(query_vector, self.__game_title_vectors)
        closest_match_index = similarity_scores.argmax()
        closest_match_game_name = self.__game_names[closest_match_index]

        spell = SpellChecker(language='en')
        words = video_game_name.split()
        corrected_words = [spell.correction(word) for word in words]
        corrected_game_name = ' '.join(corrected_words)

        closest_match = process.extractOne(corrected_game_name, self.__game_names)
        if closest_match:
            closest_match_game_name = closest_match[0]
        
        return closest_match_game_name

    def VideoGameRecommender_Genre(self, video_game_name, video_game_genre='Any'):
        default_genre = 'Any'
        video_game_idx = self.__df.query("Name == @video_game_name & Genre == @video_game_genre").index

        if video_game_idx.empty:
            video_game_idx = self.__df.query("Name == @video_game_name").index
            if not video_game_idx.empty:
                print(f"Note: Using title-based recommendations instead.")
                video_game_genre = default_genre

        if video_game_idx.empty:
            closest_match_game_name = self.VideoGameTitleRecommender(video_game_name)
            print(f"'{video_game_name}' not found. Try '{closest_match_game_name}'.")
            return
        
        video_game_list = self.__df.iloc[self.__vg_indices[video_game_idx[0]][1:]]
        if video_game_genre != default_genre:
            video_game_list = video_game_list[video_game_list['Genre'] == video_game_genre]
        
        if video_game_list.empty:
            print(f"No valid recommendations found for '{video_game_name}'.")
            return
        
        recommended_distances = np.array(self.__vg_distances[video_game_idx[0]][1:])
        if recommended_distances.size == 0:
            print("No valid distance data available.")
            return
        
        recommended_video_game_list = pd.concat([video_game_list.reset_index(drop=True), pd.DataFrame(recommended_distances, columns=['Similarity_Distance'])], axis=1)
        recommended_video_game_list[['Metascore', 'User_Score', 'Release_Year']] = recommended_video_game_list[['Metascore', 'User_Score', 'Release_Year']].fillna(0).astype(float)
        recommended_video_game_list = recommended_video_game_list.sort_values(by='Similarity_Distance', ascending=True)
        print(recommended_video_game_list.head(10))

    def VideoGameRecommender_Platform(self, video_game_name, video_game_platform='Any'):
        default_platform = 'Any'
        video_game_idx = self.__df.query("Name == @video_game_name & Platform == @video_game_platform").index

        if video_game_idx.empty:
            video_game_idx = self.__df.query("Name == @video_game_name").index
            if not video_game_idx.empty:
                print(f"Note: Using title-based recommendations instead.")
                video_game_platform = default_platform

        if video_game_idx.empty:
            closest_match_game_name = self.VideoGameTitleRecommender(video_game_name)
            print(f"'{video_game_name}' not found. Try '{closest_match_game_name}'.")
            return
        
        video_game_list = self.__df.iloc[self.__vg_indices[video_game_idx[0]][1:]]
        if video_game_platform != default_platform:
            video_game_list = video_game_list[video_game_list['Platform'] == video_game_platform]
        
        if video_game_list.empty:
            print(f"No valid recommendations found for '{video_game_name}'.")
            return
        
        recommended_distances = np.array(self.__vg_distances[video_game_idx[0]][1:])
        if recommended_distances.size == 0:
            print("No valid distance data available.")
            return
        
        recommended_video_game_list = pd.concat([video_game_list.reset_index(drop=True), pd.DataFrame(recommended_distances, columns=['Similarity_Distance'])], axis=1)
        recommended_video_game_list[['Metascore', 'User_Score', 'Release_Year']] = recommended_video_game_list[['Metascore', 'User_Score', 'Release_Year']].fillna(0).astype(float)
        recommended_video_game_list = recommended_video_game_list.sort_values(by='Similarity_Distance', ascending=True)
        print(recommended_video_game_list.head(10))

    def VideoGameRecommender(self, Type, *args):
        if Type == 'Genre':
            self.VideoGameRecommender_Genre(*args)
        elif Type == 'Platform':
            self.VideoGameRecommender_Platform(*args)
        else:
            self.VideoGameRecommender_Platform(Type)
