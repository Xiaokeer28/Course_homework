# %%
import array
import csv
import gzip
import logging
import re
import shutil
from collections import namedtuple
from pathlib import Path

from tqdm import tqdm

import click
import sentencepiece as spm
import torch

logging.basicConfig(level=logging.INFO)

MAINDIR = Path("/Users/liukeyi/Desktop/Deep Learning/student_tp7/")
DATA_PATH = MAINDIR / "data"
SRC_PATH = DATA_PATH / "training.1600000.processed.noemoticon.csv"

RE_URL = re.compile(r"(?:\@|https?\://)\S+")
RE_MENTION = re.compile(r"(?:@)\S+")
RE_NOT = re.compile(r"[^\w\s@:,;]+")


def datareader(path: Path):
    """读取 CSV，返回 (清洗后的 tweet 文本, label)"""
    with open(path, "rt", encoding="utf-8", errors="ignore") as fp:
        for row in csv.reader(fp):
            text = RE_URL.sub("", row[5])
            text = RE_MENTION.sub("@", text)
            text = RE_NOT.sub(" ", text)
            yield text, row[0]


def cleanup(src: Path, target: Path):
    if not target.is_file():
        logging.info("Creating the text data file from %s", src)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        with tmp.open("wt", encoding="utf-8") as out:
            for tweet, klass in datareader(src):
                out.write(tweet)
                out.write("\n")
        shutil.move(tmp, target)


Batch = namedtuple("Batch", ["text", "labels"])


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text: torch.LongTensor, sizes: torch.LongTensor, labels: torch.LongTensor):
        self.text = text
        self.sizes = sizes
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.text[self.sizes[index]: self.sizes[index + 1]], self.labels[index].item()

    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return Batch(
            torch.nn.utils.rnn.pad_sequence(data, batch_first=True),
            torch.LongTensor(labels),
        )


def generatedata(mode: str, tokenizer, vocab_size: int, fn: Path, label_map: dict):

    datapath = MAINDIR / f"{mode}-{vocab_size}.pth"
    logging.info("Generating dataset and saving to %s", datapath)

    text = array.array("L")
    sizes = array.array("L")
    labels = array.array("B")
    sizes.append(0)

    for tweet, label in tqdm(datareader(fn), unit=" sentences"):
        for tokenid in tokenizer.encode_as_ids(tweet):
            text.append(tokenid)
        label = int(label)
        if label in label_map:
            sizes.append(len(text))
            labels.append(label_map[label])

    data = TextDataset(
        torch.LongTensor(text),
        torch.LongTensor(sizes),
        torch.LongTensor(labels),
    )
    with gzip.open(datapath, "wb") as fp:
        torch.save(data, fp)

    logging.info(
        "Saved dataset to %s (tokens=%d, samples=%d)",
        datapath,
        len(text),
        len(labels),
    )


@click.option("--vocab-size", default=1000, type=int, help="Vocabulary size")
@click.command()
def cli(vocab_size: int):
    TRAINPATH = DATA_PATH / "sentiment140-train.txt"

    wpmodel = MAINDIR / f"wp{vocab_size}.model"
    if not wpmodel.is_file():
        logging.info("Did not find the wordpiece model %s", wpmodel)
        cleanup(SRC_PATH, TRAINPATH)
        logging.info("Création du vocabulaire avec sentencepiece")
        spm.SentencePieceTrainer.train(
            input=str(TRAINPATH),
            model_prefix=str(MAINDIR / f"wp{vocab_size}"),
            vocab_size=vocab_size,
        )
        TRAINPATH.unlink()

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(wpmodel))

    CLASSMAP = {0: 0, 4: 1}
    generatedata("train", tokenizer, vocab_size, SRC_PATH, CLASSMAP)


if __name__ == "__main__":
    cli()


# %%
