import argparse
import gzip
import json
import logging
import mlflow
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .dataset import MeliChallengeDataset
from .utils import PadSequences


logging.basicConfig(
    format="%(asctime)s: %(levelname)s - %(message)s",
    level=logging.INFO
)
# Para crear este modelo tomamos la clase MLPClassifier elaborada en la materia y cambiamos su nombre, también la estructura de la red.
# Para la estructura de la red tomamos de base la utilizada en el IMDb classifier
class CNNClassifier(nn.Module):
    def __init__(self, 
                 pretrained_embeddings_path, 
                 token_to_index,
                 filters_length,
                 n_labels,
                 hidden_layers=[256, 128],
                 vector_size=50, 
                 filters_count=200,
                 freeze_embedings=True):
        super().__init__()
        
        with gzip.open(token_to_index, "rt") as fh:
            token_to_index = json.load(fh)
        embeddings_matrix = torch.randn(len(token_to_index), vector_size)
        embeddings_matrix[0] = torch.zeros(vector_size)
        with gzip.open(pretrained_embeddings_path, "rt") as fh:
            next(fh)
            for line in fh:
                word, vector = line.strip().split(None, 1)
                if word in token_to_index:
                    embeddings_matrix[token_to_index[word]] =\
                        torch.FloatTensor([float(n) for n in vector.split()])
        self.embeddings = nn.Embedding.from_pretrained(embeddings_matrix,
                                                       freeze=freeze_embedings,
                                                       padding_idx=0)
        
        self.convs = []
        for filter_lenght in filters_length:
            self.convs.append(
                nn.Conv1d(vector_size, filters_count, filter_lenght)
            )
        self.convs = nn.ModuleList(self.convs)
        self.fc = nn.Linear(filters_count * len(filters_length), 128)
        self.output = nn.Linear(hidden_layers[-1], n_labels)
        self.vector_size = vector_size
    
    @staticmethod
    def conv_global_max_pool(x, conv):
        return F.relu(conv(x).transpose(1, 2).max(1)[0])
    
    def forward(self, x):
        x = self.embeddings(x).transpose(1, 2)  # Conv1d takes (batch, channel, seq_len)
        x = [self.conv_global_max_pool(x, conv) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = F.relu(self.fc(x))
        x = torch.sigmoid(self.output(x))
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data",
                        help="Path to the the training dataset",
                        required=True)
    parser.add_argument("--token-to-index",
                        help="Path to the the json file that maps tokens to indices",
                        required=True)
    parser.add_argument("--pretrained-embeddings",
                        help="Path to the pretrained embeddings file.",
                        required=True)
    parser.add_argument("--language",
                        help="Language working with",
                        required=True)
    parser.add_argument("--test-data",
                        help="If given, use the test data to perform evaluation.")
    parser.add_argument("--validation-data",
                        help="If given, use the validation data to perform evaluation.")
    parser.add_argument("--embeddings-size",
                        default=300,
                        help="Size of the vectors.",
                        type=int)
    parser.add_argument("--hidden-layers",
                        help="Sizes of the hidden layers of the MLP (can be one or more values)",
                        nargs="+",
                        default=[256, 128],
                        type=int)
    parser.add_argument("--dropout",
                        help="Dropout to apply to each hidden layer",
                        default=0.3,
                        type=float)
    parser.add_argument("--epochs",
                        help="Number of epochs",
                        default=3,
                        type=int)
    # Agregamos estos argumentos para poder pasarselos a run.sh directamente y no tener que estar modificando este archivo
    
    parser.add_argument("--learning_rate",
                        help="learning rate", 
                        default=0.1,
                        type=float)
    parser.add_argument("--batch_size",
                        help="batch size",
                        default=128,
                        type=int)
    parser.add_argument("--random_buffer_size",
                        help="random buffer size",
                        default=2048,
                        type=int)
    
    

    args = parser.parse_args()

    pad_sequences = PadSequences(
        pad_value=0,
        max_length=None,
        min_length=1
    )

    logging.info("Building training dataset")
    train_dataset = MeliChallengeDataset(
        dataset_path=args.train_data,
        random_buffer_size=2048  # This can be a hypterparameter
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # This can be a hyperparameter
        shuffle=False,
        collate_fn=pad_sequences,
        drop_last=False
    )

    if args.validation_data:
        logging.info("Building validation dataset")
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
        logging.info("Building test dataset")
        test_dataset = MeliChallengeDataset(
            dataset_path=args.test_data,
            random_buffer_size=1
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False
        )
    else:
        test_dataset = None
        test_loader = None

    mlflow.set_experiment(f"diplodatos.{args.language}")

    with mlflow.start_run():
        logging.info("Starting experiment")
        # Log all relevent hyperparameters
        mlflow.log_params({
            "model_type": "Convolutional Neural Network",
            "embeddings": args.pretrained_embeddings,
            "hidden_layers": args.hidden_layers,
            "dropout": args.dropout,
            "embeddings_size": args.embeddings_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate
        })
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        logging.info("Building classifier")

        model = CNNClassifier(
            pretrained_embeddings_path=args.pretrained_embeddings,
            token_to_index=args.token_to_index,
            n_labels=train_dataset.n_labels,
            hidden_layers=args.hidden_layers,
            vector_size=args.embeddings_size,
            filters_length =[4, 6, 8], 
            filters_count=200,
            freeze_embedings=True  # This can be a hyperparameter
        )
        model = model.to(device)
        loss = nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')

        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,  
            weight_decay=1e-4  
        )
        
        # optimizer = optim.SGD(
        #     model.parameters(), 
        #     lr=0.01, 
        #     momentum=0.9
        # )


        logging.info("Training classifier")
        for epoch in trange(args.epochs):
            model.train()
            running_loss = []
            for idx, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                data = batch["data"].to(device)
                target = batch["target"].to(device)
                output = model(data)
                loss_value = loss(output, target)
                loss_value.backward()
                optimizer.step()
                running_loss.append(loss_value.item())
            mlflow.log_metric("train_loss", sum(running_loss) / len(running_loss), epoch)

            if validation_dataset:
                logging.info("Evaluating model on validation")
                model.eval()
                running_loss = []
                targets = []
                predictions = []
                with torch.no_grad():
                    for batch in tqdm(validation_loader):
                        data = batch["data"].to(device)
                        target = batch["target"].to(device)
                        output = model(data)
                        running_loss.append(
                            loss(output, target).item()
                        )
                        targets.extend(batch["target"].numpy())
                        predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                    mlflow.log_metric("validation_loss", sum(running_loss) / len(running_loss), epoch)
                    mlflow.log_metric("validation_bacc", balanced_accuracy_score(targets, predictions), epoch)

        if test_dataset:
            logging.info("Evaluating model on test")
            model.eval()
            running_loss = []
            targets = []
            predictions = []
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    data = batch["data"].to(device)
                    target = batch["target"].to(device)
                    output = model(data)
                    running_loss.append(
                        loss(output, target).item()
                    )
                    targets.extend(batch["target"].numpy())
                    predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                mlflow.log_metric("test_loss", sum(running_loss) / len(running_loss), epoch)
                mlflow.log_metric("test_bacc", balanced_accuracy_score(targets, predictions), epoch)
