import argparse
import gzip
import json
import logging
import mlflow
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm, trange

from .dataset import MeliChallengeDataset
from .utils import PadSequences


logging.basicConfig(format='%(asctime)s: %(levelname)s - %(message)s', level=logging.INFO)


class RNNClassifier(nn.Module):
    def __init__(self,
                 pretrained_embeddings_path,
                 token_to_index,
                 n_labels,
                 hidden_layers=[256, 128],
                 dropout=0.3,
                 lstm_features=128,
                 lstm_layers=3,
                 vector_size=300,
                 freeze_embedings=True):
        super().__init__()
        with gzip.open(token_to_index, 'rt') as fh:
            token_to_index = json.load(fh)
        embeddings_matrix = torch.randn(len(token_to_index), vector_size)
        embeddings_matrix[0] = torch.zeros(vector_size)
        with gzip.open(pretrained_embeddings_path, 'rt') as fh:
            # Let's fill the embedding matrix
            next(fh)
            for line in fh:
                word, vector = line.strip().split(None, 1)
                if word in token_to_index:
                    wordID = token_to_index[word]
                    embeddings_matrix[wordID] = torch.FloatTensor([float(n) for n in vector.split()])
        # Embedding Layer
        self.embeddings = nn.Embedding.from_pretrained(embeddings_matrix,
                                                       freeze=freeze_embedings,
                                                       padding_idx=0)
        # LSTM Layers
        self.lstm_config = {'input_size': vector_size,
                            'hidden_size': lstm_features,
                            'num_layers': lstm_layers,
                            'batch_first': True,
                            'dropout': dropout if lstm_layers > 1 else 0.0}
        self.lstm = nn.LSTM(**self.lstm_config)
        # Linear Layers
        self.hidden_layers = [
            nn.Linear(lstm_features, hidden_layers[0])
        ]
        for input_size, output_size in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.hidden_layers.append(
                nn.Linear(input_size, output_size)
            )
        self.dropout = dropout
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output = nn.Linear(hidden_layers[-1], n_labels)
        self.vector_size = vector_size

    def forward(self, x):
        x = self.embeddings(x)
        x, _ = self.lstm(x)
        # Take last state of lstm, which is a representation of the entire text
        x = x[:, -1, :].squeeze()
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            if self.dropout:
                x = F.dropout(x, self.dropout)
        x = self.output(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Train Path
    parser.add_argument('--train-data',
                        help='Path to the training dataset.',
                        required=True)
    # Token to Index Path
    parser.add_argument('--token-to-index',
                        help='Path to the json file that maps tokens to indices.',
                        required=True)
    # Pretrained Embeddings Path
    parser.add_argument('--pretrained-embeddings',
                        help='Path to the pretrained embeddings file.',
                        required=True)
    # Language
    parser.add_argument('--language',
                        help='Language working with.',
                        required=True)
    # Test Path
    parser.add_argument('--test-data',
                        help='If given, use the test data to perform evaluation.')
    # Validation Path
    parser.add_argument('--validation-data',
                        help='If given, use the validation data to perform evaluation.')
    # Embedding Size
    parser.add_argument('--embeddings-size',
                        default=300,
                        help='Size of the vectors.',
                        type=int)
    # Hidden Layers
    parser.add_argument('--hidden-layers',
                        default=[256, 128],
                        help='Sizes of the hidden layers of the MLP (can be one or more values).',
                        nargs='+',
                        type=int)
    # Dropout
    parser.add_argument('--dropout',
                        default=0.3,
                        help='Dropout to apply to each hidden layer.',
                        type=float)
    # LSTM Features
    parser.add_argument('--lstm-features',
                        default=128,
                        help='Features of LSTM layer.',
                        type=int)
    # LSTM Layers
    parser.add_argument('--lstm-layers',
                        default=3,
                        help='Ammount of LSTM layers.',
                        type=int)
    # Epochs
    parser.add_argument('--epochs',
                        default=3,
                        help='Number of epochs.',
                        type=int)
    # Batch Size
    parser.add_argument('--batch-size',
                        default=128,
                        help='Size of the batch.',
                        type=int)
    # Freeze Embeddings
    parser.add_argument('--freeze-embeddings',
                        default=True,
                        help='Freeze embeddings during training.',
                        type=bool)
    # Learning Rate
    parser.add_argument('--learning-rate',
                        default=1e-3,
                        help='Learning rate.',
                        type=float)
    # Weight Decay
    parser.add_argument('--weight-decay',
                        default=1e-5,
                        help='Weight decay.',
                        type=float)
    # Random Buffer Size
    parser.add_argument('--random-buffer-size',
                        default=2048,
                        help='Size of the randomizer buffer.',
                        type=int)

    args = parser.parse_args()

    pad_sequences = PadSequences(
        pad_value=0,
        max_length=None,
        min_length=1
    )

    logging.info('Building training dataset')
    train_dataset = MeliChallengeDataset(
        dataset_path=args.train_data,
        random_buffer_size=args.random_buffer_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_sequences,
        drop_last=False
    )

    if args.validation_data:
        logging.info('Building validation dataset')
        validation_dataset = MeliChallengeDataset(
            dataset_path=args.validation_data,
            random_buffer_size=1
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False
        )
    else:
        validation_dataset = None
        validation_loader = None

    if args.test_data:
        logging.info('Building test dataset')
        test_dataset = MeliChallengeDataset(
            dataset_path=args.test_data,
            random_buffer_size=1
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False
        )
    else:
        test_dataset = None
        test_loader = None

    mlflow.set_experiment(f'diplodatos.{args.language}')

    with mlflow.start_run():
        logging.info('Starting experiment')
        # Log all relevent hyperparameters
        mlflow.log_params({
            'model_type': 'Recurrent Neural Network',
            'embeddings': args.pretrained_embeddings,
            'hidden_layers': args.hidden_layers,
            'dropout': args.dropout,
            'lstm_features': args.lstm_features,
            'lstm_layers': args.lstm_layers,
            'embeddings_size': args.embeddings_size,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'freeze_embeddings': args.freeze_embeddings,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'random_buffer_size': args.random_buffer_size
        })
        # Look for the device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        logging.info('Building classifier')
        model = RNNClassifier(
                pretrained_embeddings_path=args.pretrained_embeddings,
                token_to_index=args.token_to_index,
                n_labels=train_dataset.n_labels,
                hidden_layers=args.hidden_layers,
                dropout=args.dropout,
                lstm_features=args.lstm_features,
                lstm_layers=args.lstm_layers,
                vector_size=args.embeddings_size,
                freeze_embedings=args.freeze_embeddings
                )
        # Send the model to the device
        model = model.to(device)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        logging.info('Training classifier')
        for epoch in trange(args.epochs):
            model.train()
            running_loss = []
            for idx, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                data = batch['data'].to(device)
                target = batch['target'].to(device)
                output = model(data)
                loss_value = loss(output, target)
                loss_value.backward()
                optimizer.step()
                running_loss.append(loss_value.item())
            mlflow.log_metric('train_loss', sum(running_loss) / len(running_loss), epoch)

            if validation_dataset:
                logging.info('Evaluating model on validation')
                model.eval()
                running_loss = []
                targets = []
                predictions = []
                with torch.no_grad():
                    for batch in tqdm(validation_loader):
                        data = batch['data'].to(device)
                        target = batch['target'].to(device)
                        output = model(data)
                        loss_value = loss(output, target)
                        running_loss.append(loss_value.item())
                        targets.extend(batch['target'].numpy())
                        predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                    mlflow.log_metric('validation_loss', sum(running_loss) / len(running_loss), epoch)
                    mlflow.log_metric('validation_bacc', balanced_accuracy_score(targets, predictions), epoch)

        if test_dataset:
            logging.info('Evaluating model on test')
            model.eval()
            running_loss = []
            targets = []
            predictions = []
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    data = batch['data'].to(device)
                    target = batch['target'].to(device)
                    output = model(data)
                    loss_value = loss(output, target)
                    running_loss.append(loss_value.item())
                    targets.extend(batch['target'].numpy())
                    predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                mlflow.log_metric('test_loss', sum(running_loss) / len(running_loss), epoch)
                mlflow.log_metric('test_bacc', balanced_accuracy_score(targets, predictions), epoch)