import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split,TensorDataset
from pathlib import Path
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

BATCH_SIZE = 311
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
LOG_PATH = "/tmp/runs/lightning_logs"

# Changer le DATA_PATH
DATA_PATH = "/tmp/mnist"

from sklearn.datasets import fetch_openml

class Lit3LayerMLP(pl.LightningModule):
    def __init__(
        self,
        dim_in,
        dim_hidden=100,
        dim_out=10,
        learning_rate=1e-3,
        p_dropout=0.0,
        weight_decay=0.0,   # L2 via weight decay
    ):
        super().__init__()
        self.save_hyperparameters()

        # Réseau : 3 couches cachées de 100 neurones + couche de sortie
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_hidden)
        self.classifier = nn.Linear(dim_hidden, dim_out)

        self.dropout = nn.Dropout(p_dropout)
        self.loss = nn.CrossEntropyLoss()
        self.name = "mnist-3layers"

        # pour accumuler les sorties
        self.training_outputs = []
        self.valid_outputs = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        # L2 via weight decay
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        acc = (yhat.argmax(1) == y).sum()

        logs = {"loss": loss, "accuracy": acc, "nb": len(x)}
        self.log("accuracy", acc / len(x), on_step=False, on_epoch=True)
        self.training_outputs.append(logs)
        return logs

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        acc = (yhat.argmax(1) == y).sum()

        logs = {"loss": loss, "accuracy": acc, "nb": len(x)}
        self.log("val_accuracy", acc / len(x), on_step=False, on_epoch=True)
        self.valid_outputs.append(logs)
        return logs

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        acc = (yhat.argmax(1) == y).sum()
        logs = {"loss": loss, "accuracy": acc, "nb": len(x)}
        return logs

    def log_x_end(self, outputs, phase):
        total_acc = sum([o["accuracy"] for o in outputs])
        total_nb = sum([o["nb"] for o in outputs])
        total_loss = sum([o["loss"] for o in outputs]) / len(outputs)
        total_acc = total_acc / total_nb
        self.log_dict({f"loss/{phase}": total_loss, f"acc/{phase}": total_acc})

    def on_training_epoch_end(self):
        self.log_x_end(self.training_outputs, "train")
        self.training_outputs.clear()

    def on_validation_epoch_end(self):
        self.log_x_end(self.valid_outputs, "valid")
        self.valid_outputs.clear()

    def on_test_epoch_end(self):
        pass



class LitMnistData(pl.LightningDataModule):

    def __init__(self,batch_size=BATCH_SIZE,train_ratio=TRAIN_RATIO,test_ratio=TEST_RATIO):
        super().__init__()
        self.dim_in = None
        self.dim_out = None
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    def prepare_data(self):
        ### Do not use "self" here.
        # S'il faut charger d'abord les données ou autres.
        pass

    def setup(self,stage=None):
        x,y = fetch_openml("mnist_784",return_X_y=True,as_frame=False, data_home=DATA_PATH)
        x = torch.tensor(x).view(-1,x.shape[1]).float()/255.
        y = torch.tensor(y.astype(int)).long()
        train_length = int(x.shape[0]*self.train_ratio)
        test_length = int(x.shape[0]*self.test_ratio)
        val_length = x.shape[0]-train_length-test_length
        if stage =="fit" or None:
            # Si on est en phase d'apprentissage ou val
            x,y  = x[:x.shape[0]-test_length], y[:x.shape[0]-test_length]
            self.dim_in = x.shape[1]
            self.dim_out = len(set(y))
            ds_train = TensorDataset(x,y)
            self.mnist_train, self.mnist_val, = random_split(ds_train,[train_length,x.shape[0]-train_length])
        if stage == "test":
            # en phase de test
            x,y = x[x.shape[0]-test_length:],y[x.shape[0]-test_length:]
            self.mnist_test= TensorDataset(x,y)

    def train_dataloader(self):
        return DataLoader(self.mnist_train,batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.mnist_val,batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.mnist_test,batch_size=self.batch_size)


data = LitMnistData()

# Doivent etre appelées manuellement ici, car on a besoin de dim_in. Sinon, l'appel est automatique
data.prepare_data()
data.setup(stage="fit")

model = Lit3LayerMLP(
    dim_in=data.dim_in,
    dim_hidden=100,
    dim_out=data.dim_out,
    learning_rate=1e-3,
    p_dropout=0.2,       
    weight_decay=1e-4,
)   


logger = TensorBoardLogger(save_dir=LOG_PATH,name=model.name,version=time.asctime(),default_hp_metric=False)

trainer = pl.Trainer(default_root_dir=LOG_PATH,logger=logger,max_epochs=100)
trainer.fit(model,datamodule=data)
trainer.test(model,datamodule=data)
