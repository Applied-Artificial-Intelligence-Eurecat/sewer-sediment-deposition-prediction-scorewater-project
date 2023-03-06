from flask import Flask
from flask import request
import torch
import torch.nn as nn
from joblib import load
import pandas as pd

from src.deploy.models import AE, NeuralNetwork, arguments_assignment

app = Flask(__name__)


@app.post("/predict")
def predict_sediment_level():
    '''
    A post with the model vector to predict
    /predict and the body is key values, with key being:
    {'section', 'pipeheight', 'pipewidth', 'perimeter', 'Length', 'Velocity',
       'waterheight', 'flow', 'section_1', 'pipeheight_1', 'pipewidth_1', 'perimeter_1',
       'Length_1', 'Velocity_1', 'waterheight_1', 'flow_1', 'section_2',
       'pipeheight_2', 'pipewidth_2', 'perimeter_2', 'Length_2', 'Velocity_2',
       'waterheight_2', 'flow_2', 'section_3', 'pipeheight_3', 'pipewidth_3',
       'perimeter_3', 'Length_3', 'Velocity_3', 'waterheight_3', 'flow_3',
       'section_4', 'pipeheight_4', 'pipewidth_4', 'perimeter_4', 'Length_4',
       'Velocity_4', 'waterheight_4', 'flow_4', 'section_5', 'pipeheight_5',
       'pipewidth_5', 'perimeter_5', 'Length_5', 'Velocity_5', 'waterheight_5',
       'flow_5', 'neighbourhood', 'amount_rain_mean', 'amount_rain_std', 'value_0', 'value_1',
       'cleaning_applied_0', 'cleaning_applied_1', 'amount_rain_mean_1',
       'amount_rain_std_1', 'value_0_1', 'value_1_1', 'cleaning_applied_0_1',
       'cleaning_applied_1_1', 'amount_rain_mean_2', 'amount_rain_std_2',
       'value_0_2', 'value_1_2', 'cleaning_applied_0_2',
       'cleaning_applied_1_2', 'amount_rain_mean_3', 'amount_rain_std_3',
       'value_0_3', 'value_1_3', 'cleaning_applied_0_3',
       'cleaning_applied_1_3', 'amount_rain_mean_4', 'amount_rain_std_4',
       'value_0_4', 'value_1_4', 'cleaning_applied_0_4',
       'cleaning_applied_1_4', 'amount_rain_mean_5', 'amount_rain_std_5',
       'value_0_5', 'value_1_5', 'cleaning_applied_0_5',
       'cleaning_applied_1_5'}
    '''
    pvector, dvector = format_values(request.form)
    models = load_models()
    result = predict_probabilities(models, pvector, dvector)
    return structure_response(result)


def format_values(form_data):
    '''
    Converts arguments into torch vector
    '''
    form_data = form_data.to_dict()
    pvector = []
    dvector = []
    # scale the data
    for k in form_data.keys():
        if arguments_assignment[k] == 'pvector':
            pvector.append(int(form_data[k]))
        else:
            dvector.append(int(form_data[k]))
    aescaler = load('models_state_dict/AEscaler.joblib')
    pvector = aescaler.transform([pvector])[0]
    return pvector, dvector


def load_models():
    ae = AE()
    sequence = nn.Sequential(nn.Linear(46, 10), nn.Softsign(), nn.Linear(10, 10), nn.Softsign(), nn.Linear(10, 1),
                             nn.Sigmoid())
    ann5 = NeuralNetwork(sequence)
    ann10 = NeuralNetwork(sequence)
    ann15 = NeuralNetwork(sequence)
    ann20 = NeuralNetwork(sequence)
    ae.load_state_dict(torch.load('models_state_dict/AE'))
    ann5.load_state_dict(torch.load('models_state_dict/model_threshold_5'))
    ann10.load_state_dict(torch.load('models_state_dict/model_threshold_10'))
    ann15.load_state_dict(torch.load('models_state_dict/model_threshold_15'))
    ann20.load_state_dict(torch.load('models_state_dict/model_threshold_20'))
    return [ae, ann5, ann10, ann15, ann20]


def predict_probabilities(models, pvector, dvector):
    device = torch.device('cpu')
    for model in models:
        model.to(device)
        model.eval()
    pvector = torch.tensor(pvector, dtype=torch.float32)
    pvector = pvector.to(device)  # Run forward pass
    with torch.no_grad():
        pred = models[0].encode(pvector)
        pred = pred.tolist()
        dvector = dvector + pred + [0]
        annscaler = load('models_state_dict/ANNscaler.joblib')
        dvector = annscaler.transform([dvector])[0][:-1]
        dvector = torch.tensor(dvector, dtype=torch.float32)
        dvector = dvector.to(device)
        result = [True if torch.round(model(dvector)) else False for model in models[1:]]
    return result


def structure_response(response):
    return {'Occupation of more than 5%': response[0],
            'Occupation of more than 10%': response[1],
            'Occupation of more than 15%': response[2],
            'Occupation of more than 20%': response[3]}


@app.post("/neighbourhood")
def find_best_neighbours():
    '''
    POST with params:
    id: int
    file: csv file
    '''
    pipe_id = int(request.form['id'])
    file = request.files['file']
    df = pd.read_csv(file)
    return get_similars(df.loc[df['Id'] == pipe_id], df).to_dict()


def get_similars(x, df):
    return get_top_adjacents(calculate_punctuations(x, get_close_arcs(x, df)))


# Careful, some data might not have close arcs and this function will break
def get_close_arcs(x, df):
    df_ = pd.DataFrame()
    max = 200
    while len(df_) == 0:
        df_ = df.loc[((df.beginning_point_x - x.iloc[0].beginning_point_x).abs().between(0, max)) & (
                (df.beginning_point_y - x.iloc[0].beginning_point_y).abs().between(0, max) & (~df.Id.isin(x.Id)))]
        max = max + 100
    return df_


def calculate_punctuations(arc, df):
    return df.groupby('Id').apply(lambda x: calculate_punctuation(arc, x))


def calculate_punctuation(arc, adjacent_arc):
    adjacent_arc['punctuation'] = abs(arc.value[1:].mean() - adjacent_arc.value[1:].mean())
    return adjacent_arc


def get_top_adjacents(df):
    return df.drop_duplicates(['Id']).sort_values(by='punctuation', ascending=True).head(5).Id
