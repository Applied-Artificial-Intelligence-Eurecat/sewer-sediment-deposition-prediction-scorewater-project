import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from src.models.to_github.models.ann import NeuralNetwork
from src.models.to_github.models.autoencoder import AutoEncoder

config = {}


def main():
    df = load_data()
    model, scaler = train_AE(df)
    df = generate_datamodel(df, model, scaler)
    second_method_training(df)


def load_data():
    df = pd.read_csv(config['data']['path'])
    return df


def train_AE(df):
    ae_df = df.drop(df.filter(regex=("(value|amount_rain|cleaning_applied|days_between).*")).columns, axis=1)
    ae_df.drop_duplicates(inplace=True)
    # normalize min max
    scaler = MinMaxScaler()
    ae_df.iloc[:] = scaler.fit_transform(ae_df.iloc[:])
    train = torch.tensor(ae_df.values.tolist(), dtype=torch.float32)
    batch_size = 60
    train_data = TensorDataset(train, train)
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=False, shuffle=True)
    torch.manual_seed(1)
    device = torch.device("cpu")
    # create a model from `AutoEncoder` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AutoEncoder(ae_df.shape[1], config['AE']['layer1'], config['AE']['layer2'],
                        config['AE']['layer3'], config['AE']['activation']).to(device)
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # error loss
    criterion = nn.MSELoss()
    epochs = config['AE']['epochs']
    epoch_results = []
    test_loss = []
    for epoch in range(epochs):
        avg_loss = 0
        for batch_features, _ in train_loader:
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features.to(device))

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            avg_loss += train_loss.item()

        # display the epoch training loss
        # print("Epoch {}/{} Done, Total Loss: {}".format(epoch + 1, epochs, avg_loss / len(train_loader)))
        epoch_results.append(avg_loss / len(train_loader))
    model.eval()
    print("AE end")
    return model, scaler


def generate_datamodel(df, model, scaler):
    df_encoded = encode_physical_parameters(df, model, scaler)
    scaler = MinMaxScaler()
    df.loc[:, df.columns != 'value_2'] = scaler.fit_transform(df.loc[:, df.columns != 'value_2'])
    df = df[df.filter(regex=("(value|amount_rain|cleaning_applied|days_between).*")).columns].copy()
    # The conditional will decide the type of sediment occupation threshold
    df['value_y'] = (df['value_2'] >= 5).astype('int')
    df.drop('value_2', axis=1, inplace=True)
    df.drop('cleaning_applied_2', axis=1, inplace=True)
    df.drop(df.filter(regex=("value_2_.")).columns, axis=1, inplace=True)
    df.drop(df.filter(regex=("cleaning_applied_2_.")).columns, axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, df_encoded], axis=1)
    return df


def encode_physical_parameters(df, model, scaler):
    physical_features = df.drop(df.filter(regex=("(value|amount_rain|cleaning_applied|days_between).*")).columns,
                                axis=1)
    physical_features.iloc[:] = scaler.transform(physical_features.iloc[:])
    physical_features = torch.tensor(physical_features.values.tolist(), dtype=torch.float32)
    with torch.no_grad():
        encoded_df = model.encode(physical_features)
    df_ = pd.DataFrame(encoded_df.detach().numpy())
    return df_


def second_method_training(df):
    train_loaders, test_loaders, weights = generate_train_test(df)
    optimize_and_predict(df, test_loaders, train_loaders, weights)


def generate_train_test(df):
    batch_size = 32
    train_loaders = []
    test_loaders = []
    weights = []
    fold = KFold(n_splots=5)
    # generate loaders
    for i, (train_index, test_index) in enumerate(fold.split(df)):
        X_train = torch.tensor(df.loc[df.index[train_index]].drop('value_y').values.tolist(), dtype=torch.float32)
        y_train = torch.tensor(df.loc[df.index[train_index], 'value_y'].values.tolist(), dtype=torch.float32)
        X_test = torch.tensor(df.loc[df.index[train_index]].drop('value_y').values.tolist(), dtype=torch.float32)
        y_test = torch.tensor(df.loc[df.index[train_index], 'value_y'].values.tolist(), dtype=torch.float32)
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False)
        train_loaders.append(train_loader)
        test_data = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=len(y_test), drop_last=False)
        test_loaders.append(test_loader)
        # Adding weights to overcome imbalance. Depending on the dataset, it needs to change.
        weights.append(((len(y_train) - torch.count_nonzero(y_train)) / torch.count_nonzero(y_train)) * 2.1)
    return train_loaders, test_loaders, weights


def optimize_and_predict(df, test_loaders, train_loaders, weights):
    nn_architecture = nn.Sequential(nn.Linear(df.shape[1] - 1, config['ANN']['layer1']), config['ANN']['activation'],
                                    nn.Linear(config['ANN']['layer1'], config['ANN']['layer2']),
                                    config['ANN']['activation'],
                                    nn.Linear(config['ANN']['layer2'], 1),
                                    nn.Sigmoid())
    results = []
    for train_loader, test_loader, weight in zip(train_loaders, test_loaders, weights):
        results.append(
            train_ann(train_loader, test_loader, nn_architecture, epochs=config['ANN']['epochs'], pos_weight=weight))
    # Save scores
    tnr50 = sum([result['50-TNR'] for result in results]) / len(results)
    tpr50 = sum([result['50-recall'] for result in results]) / len(results)
    accuracy50 = sum([result['50-accuracy'] for result in results]) / len(results)
    with open(config['results']['path'] + '/multiple_execs.txt', 'a') as f:
        f.write(
            '''AE architecture: Layer 1 {}, Layer 2 {}, Layer 3 {}, Activation function {}, Epochs {}
            ANN architecture: Layer 1 {}, Layer 2 {}, Activation function {}, Epochs {}
            Scores: TNR {}, TPR {}, ACC {}'''.format(
                config['AE']['layer1'], config['AE']['layer2'], config['AE']['layer3'],
                config['AE']['activation'], config['AE']['epochs'],
                config['ANN']['layer1'], config['ANN']['layer2'],
                config['ANN']['activation'], config['ANN']['epochs'],
                tnr50, tpr50, accuracy50
            )
        )


def train_ann(train_loader, test_loader, sequence, epochs=500, pos_weight=1):
    torch.manual_seed(42)
    device = torch.device("cpu")
    # load it to the specified device, either gpu or cpu
    model = NeuralNetwork(sequence=sequence).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # BCE loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    epoch_results = []
    evaluations = {}
    for epoch in range(epochs):
        avg_loss = 0
        for x, y_train in train_loader:
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(x.to(device))

            # compute training reconstruction loss
            loss = criterion(outputs, y_train.unsqueeze(1))

            # compute accumulated gradients
            loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            avg_loss += loss.item()

        if (epoch + 1) % 50 == 0:  # 0 to 499 +1 -> 1 to 500 looping
            model.eval()
            evaluations[str(epoch + 1) + '-TNR'], \
            evaluations[str(epoch + 1) + '-recall'], \
            evaluations[str(epoch + 1) + '-accuracy'], \
            evaluations[str(epoch + 1) + '-AUC'], \
            evaluations[str(epoch + 1) + '-precision'], \
            evaluations[str(epoch + 1) + '-cm'] = evaluate(model, test_loader)
            model.train()

        epoch_results.append(avg_loss / len(train_loader))
    model.eval()
    return evaluations


def evaluate(model, test_set):
    y_pred_list = []
    y_test_list = []
    device = torch.device("cpu")
    with torch.no_grad():
        for x_test, y_test in test_set:
            out = model(x_test.to(device))
            out = torch.round(out)
            y_pred_list.append(out.numpy())
            y_test_list.append(y_test.numpy())

    y_pred = [a.squeeze().tolist() for a in y_pred_list][0]
    y_test = [a.squeeze().tolist() for a in y_test_list][0]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    area = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tnr = cm[0][0] / (cm[0][0] + cm[0][1])
    return tnr, recall, accuracy, area, precision, cm


def load_config():
    with open('conf.yaml') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':
    config = load_config()
    main()
