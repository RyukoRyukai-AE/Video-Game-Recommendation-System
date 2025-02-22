import yaml
import joblib
from os.path import join
from Game_Recommender import Recommender
import pandas as pd

config_path = "D:/Workspace/Machine_Learning/Recommendation_System/Game_Recommendation_System/src/config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

data_path = config['paths']['data']
model_path = config['paths']['model']

vectorizer = joblib.load(join(model_path, 'Vectorizer.pkl'))
nn_model = joblib.load(join(model_path, 'Nearest_Neighbor.pkl'))

scaled_features = pd.read_csv(join(data_path, 'processed/scaled_features.csv'))
df = pd.read_csv(join(data_path, 'processed/processed_data.csv'))

game_names = df['Name'].drop_duplicates()
game_names = game_names.reset_index(drop=True)

rec1 = Recommender(vectorizer=vectorizer, game_names=game_names, nn_model=nn_model, scaled_features=scaled_features, df=df)

# Test case
rec1.VideoGameRecommender('NieR: Automata')