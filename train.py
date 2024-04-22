from spacy.lang.en import English, Language
from gensim.models import Word2Vec
from datasets import load_dataset, Dataset, config

import functools
import logging
import os

from typing import Iterable

# from https://discuss.python.org/t/string-isplit-iterator-based-split-for-strings/7533/15
def isplit(s: str, sep: str):
    """Lazy version of s.split(sep)

    >>> list(isplit("", ","))
    ['']
    >>> list(isplit("AAA", ","))
    ['AAA']
    >>> list(isplit("AAA,", ","))
    ['AAA', '']
    >>> list(isplit("AAA,BBB", ","))
    ['AAA', 'BBB']
    >>> list(isplit("AAA,,BBB", ",,"))
    ['AAA', 'BBB']
    """
    seplen = len(sep)
    if seplen == 0:
        raise ValueError("empty separator")

    start = 0
    while True:
        index = s.find(sep, start)
        if index == -1:
            yield s[start:]
            return
        yield s[start:index]
        start = index + seplen


class SentencesBase:
    """Base class for sentence iterators."""
    def __init__(self, pipeline: Language):
        self.pipeline = pipeline

    def get_sentences(self, texts: Iterable[str]) -> Iterable[list[list[str]]]:
        """Split the specified texts into sentences, consisting of text tokens."""
        for doc in self.pipeline.pipe(texts):
            for sent in doc.sents:
                sent = [token.text.lower() for token in sent if not token.is_space]
                if len(sent) == 0:
                    continue
                yield sent


class YearFileSentences(SentencesBase):
    """Iterate over the sentences in the corresponding text
    files in the specified corpus."""
    def __init__(self, dirname: str, pipeline: Language, start_year: int, end_year: int):
        super().__init__(pipeline)
        self.dirname = dirname
        self.start_year = start_year
        self.end_year = end_year

    def __iter__(self):
        self.num_tokens = 0
        for i in range(self.start_year, self.end_year + 1):
            file_path = os.path.join(self.dirname, f"{i}.txt")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="latin1") as f:
                    for sent in self.get_sentences(f):
                        yield sent
            else:
                logging.warning(f"{i}.txt not found in {self.dirname}")


class DatasetSentences(SentencesBase):
    """Iterate over the sentences in the corresponding splits in the
    specified corpus."""
    def __init__(self, dataset: Dataset, pipeline: Language, start_year: int, end_year: int):
        super().__init__(pipeline)
        self.dataset = dataset
        self.start_year = start_year
        self.end_year = end_year

    def __iter__(self):
        self.num_tokens = 0
        for i in range(self.start_year, self.end_year + 1):
            if str(i) in self.dataset:
                for sent in self.get_sentences(self.dataset[str(i)]["article"]):
                    yield sent
            else:
                logging.warn(f"{i} not found in dataset")


class SentenceMixer:
    """"Mixes sentences from multiple iterables, interleaving them and
    balancing the number of sentences from each iterable."""
    def __init__(self, *sentence_iterables: list[SentencesBase]):
        self.sentence_iterables = sentence_iterables

    def __iter__(self):
        iterators = [iter(sentence_iterable) for sentence_iterable in self.sentence_iterables]
        finished_idxs = set()
        while len(finished_idxs) < len(self.sentence_iterables):
            for i, iterator in enumerate(iterators):
                try:
                    yield next(iterator)
                except StopIteration:
                    finished_idxs.add(i)
                    iterators[i] = iter(self.sentence_iterables[i])


MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

english = English(pipeline=[], batch_size=100)
sentencizer = english.add_pipe("sentencizer")

config.IN_MEMORY_MAX_SIZE = 1e9
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "all_years",
    trust_remote_code=True,
)

constructors = {
    "ussal": functools.partial(YearFileSentences, "US-SAL-Corpus/text", english),
    "usr": functools.partial(YearFileSentences, "US-R-Corpus/text", english),
    # "as": functools.partial(DatasetSentences, dataset, english),
}

model_tag = "_".join(constructors.keys())

partition_starts = [1770] + list(range(1800, 1980, 10))

if __name__ == "__main__":
    for i in range(len(partition_starts) - 1):
        start_year = partition_starts[i]
        end_year = partition_starts[i + 1] - 1

        logging.info(f"Training a model for {start_year}-{end_year}...")

        sentence_iterables = [constructor(start_year, end_year) for constructor in constructors.values()]
        sentences = SentenceMixer(*sentence_iterables)

        # params from https://github.com/williamleif/histwords/blob/master/sgns/runword2vec.py
        model = Word2Vec(
            sentences=sentences,
            vector_size=300,
            window=2,
            workers=6,
            sg=1,
            hs=0,
            sample=0,
            negative=5,
            min_count=100
        )

        model.save(os.path.join(MODELS_DIR, f"model_{model_tag}_{start_year}-{end_year}.model"))
