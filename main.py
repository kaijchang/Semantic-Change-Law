from spacy.lang.en import English, Language
from gensim.models import KeyedVectors, Word2Vec
from datasets import load_dataset, Dataset, config

import logging
import os

from typing import Iterable

english = English(pipeline=[], batch_size=100)
sentencizer = english.add_pipe("sentencizer")

config.IN_MEMORY_MAX_SIZE = 1e9
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "all_years",
    trust_remote_code=True,
)

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
        self.num_tokens = 0

    def get_sentences(self, texts: Iterable[str]) -> Iterable[list[list[str]]]:
        """Split the specified texts into sentences, consisting of text tokens."""
        for doc in self.pipeline.pipe(texts):
            for sent in doc.sents:
                sent = [token.text.lower() for token in sent if not token.is_space]
                self.num_tokens += len(sent)
                yield sent


class YearFileSentences(SentencesBase):
    """Iterate over the sentences in the corresponding text
    files in the specified corpus."""
    def __init__(self, dirname: str, start_year: int, end_year: int, pipeline: Language):
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
                logging.warn(f"{i}.txt not found in {self.dirname}")


class DatasetSentences(SentencesBase):
    """Iterate over the sentences in the corresponding splits in the
    specified corpus."""
    def __init__(self, dataset: Dataset, start_year: int, end_year: int, pipeline: Language):
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

# for sentence in YearFileSentences("US-R-Corpus/text", 1770, 1799, english):
#     print(sentence)

for sentence in YearFileSentences("US-SAL-Corpus/text", 1770, 1799, english):
    print(sentence)

# for sentence in DatasetSentences(dataset, 1770, 1799, english):
#     print(sentence)
