#!/usr/bin/env python3

from spacy.lang.en import English, Language
from datasets import Dataset, load_dataset, config

import functools
import logging
import os
import re

from typing import Iterable

MAX_TOKENS = 2.5e7
SEED = 42
MODELS_DIR = "models"


# from https://discuss.python.org/t/string-isplit-iterator-based-split-for-strings/7533/15
def isplit(s: str, seps: list[str]):
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
    seplens = [len(sep) for sep in seps]
    if any(seplen == 0 for seplen in seplens):
        raise ValueError("empty separator")

    start = 0
    while True:
        next_idxs = [s.find(sep, start) for sep in seps]
        index = (
            min(idx for idx in next_idxs if idx != -1)
            if any(idx != -1 for idx in next_idxs)
            else -1
        )
        if index == -1:
            yield s[start:]
            return
        yield s[start:index]
        start = index + seplens[next_idxs.index(index)]


class SentencesBase:
    """Base class for sentence iterators."""

    def __init__(self, pipeline: Language, max_tokens: int = MAX_TOKENS):
        self.pipeline = pipeline
        self.num_tokens = 0
        self.max_tokens = max_tokens

    def clean_text(self, text: str) -> str:
        return text

    def get_sentences(self, texts: Iterable[str]) -> Iterable[list[list[str]]]:
        """Split the specified texts into sentences, consisting of text tokens."""
        for doc in self.pipeline.pipe(self.clean_text(text) for text in texts):
            for sent in doc.sents:
                sent = [
                    token.text.lower() if not token.like_num else "<NUM>"
                    for token in sent
                    if token.is_alpha or token.like_num
                ]
                if len(sent) == 0:
                    continue
                yield sent
                self.num_tokens += len(sent)
                if self.num_tokens >= self.max_tokens:
                    return

    def __iter__(self):
        self.num_tokens = 0


class YearFileSentences(SentencesBase):
    """Iterate over the sentences in the corresponding text
    files in the specified corpus."""

    def __init__(
        self,
        dirname: str,
        pipeline: Language,
        start_year: int,
        end_year: int,
        max_tokens: int = MAX_TOKENS,
    ):
        super().__init__(pipeline, max_tokens)
        self.dirname = dirname
        self.start_year = start_year
        self.end_year = end_year

    def __iter__(self):
        super().__iter__()
        for i in range(self.start_year, self.end_year + 1):
            file_path = os.path.join(self.dirname, f"{i}.txt")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="latin1") as f:
                    for sent in self.get_sentences(
                        isplit(re.sub(r"-\.?\n+", "", f.read()), ["\f", "\n\n"])
                    ):
                        yield sent
            else:
                logging.warning(f"{i}.txt not found in {self.dirname}")


class DatasetSentences(SentencesBase):
    """Iterate over the sentences in the corresponding splits in the
    specified corpus."""

    def __init__(
        self,
        dataset: Dataset,
        pipeline: Language,
        start_year: int,
        end_year: int,
        max_tokens: int = MAX_TOKENS,
    ):
        super().__init__(pipeline, max_tokens)
        self.dataset = dataset
        self.start_year = start_year
        self.end_year = end_year
        self.iterable_splits = {}

    def __iter__(self):
        super().__iter__()
        self.num_tokens = 0
        for i in range(self.start_year, self.end_year + 1):
            if str(i) in self.dataset:
                split: Dataset = self.dataset[str(i)]
                if i not in self.iterable_splits:
                    self.iterable_splits[i] = split.to_iterable_dataset(
                        num_shards=min(128, split.num_rows)
                    ).shuffle(SEED)
                split = self.iterable_splits[i]
                for sent in self.get_sentences(row["article"] for row in split):
                    yield sent
            else:
                logging.warn(f"{i} not found in dataset")


class SentenceMixer:
    """ "Mixes sentences from multiple iterables, interleaving them and
    balancing the number of sentences from each iterable."""

    def __init__(self, *sentence_iterables: list[SentencesBase]):
        self.sentence_iterables = sentence_iterables

    def __iter__(self):
        iterators = [
            iter(sentence_iterable) for sentence_iterable in self.sentence_iterables
        ]
        finished_idxs = set()
        while len(finished_idxs) < len(self.sentence_iterables):
            for i, iterator in enumerate(iterators):
                try:
                    yield next(iterator)
                except StopIteration:
                    finished_idxs.add(i)
                    iterators[i] = iter(self.sentence_iterables[i])


# english = spacy.load("en_core_web_sm", disable=["ner", "parser"])
english = English(pipeline=[], batch_size=100)
sentencizer = english.add_pipe("sentencizer")

config.IN_MEMORY_MAX_SIZE = 1e9
dataset = load_dataset(
    "dell-research-harvard/AmericanStories",
    "all_years",
    trust_remote_code=True,
)

constructors = {
    "ussal": functools.partial(YearFileSentences, "US-SAL-Corpus/text", english),
    "usr": functools.partial(YearFileSentences, "US-R-Corpus/text", english),
    "as": functools.partial(DatasetSentences, dataset, english),
}

os.makedirs(MODELS_DIR, exist_ok=True)

PARTITION_STARTS = list(range(1770, 1980, 10))
VOCAB_CUTOFF_YEAR = 1800
MIN_COUNT = 50


def get_model_tag():
    model_constructors_keys = [key for key in constructors.keys()]
    return (
        model_constructors_keys,
        f"{'_'.join(model_constructors_keys)}_{PARTITION_STARTS[0]}-{PARTITION_STARTS[-1] - 1}_{VOCAB_CUTOFF_YEAR}",
    )
