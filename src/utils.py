"""MLW-OCR Projekt.

File for utilty functions.
"""
from typing import Any, Callable, Dict, Tuple

import logging
import numpy as np
import torch
import copy
from PIL import Image, ImageEnhance
from torchvision import transforms
import functools
import typing

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, DatasetDict, load_dataset, Dataset
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
# from torchmetrics import CharErrorRate
from torchmetrics.text import CharErrorRate
from transformers import PreTrainedTokenizerFast


def get_dataset(dataset_identifier: str, processor: Any, tokenizer: PreTrainedTokenizerFast, augmentation: bool = False, aug_incr: int = 4,debug: bool = False) -> DatasetDict:
    """Get Dataset.

    Download and pre-process data. Truncate for debugging if `debug` is True.
    
    :param dataset_identifier: Dataset identifier for huggingface datasets.
    :param processor: (TrOCR) pocessor for image data.
    :param tokenizer: Tokenizer for text data. # TODO Add type
    :param augmentation: Augmentation for image data.
    :param aug_incr: Increases the data `aug_incr`-fold if augmentation is True.
    :param debug: Debug flag.

    :returns: Pre-processed dataset.
    """
    dataset: DatasetDict = load_dataset(dataset_identifier)
    dataset = dataset.map(lambda e: {'image': e['image'] if e['image'].mode == 'RGB' else e['image'].convert('RGB'), 'text': e['text']})
    if debug:
        logging.info("Truncate dataset to 30 samples.")
        dataset = DatasetDict({k:Dataset.from_dict(dict(v[0:3])) for k,v in dataset.items()})
    if augmentation:
        logging.info("Apply Augmentation")
        logging.info("Augmentation will increase the dataset by a factor of %d.", aug_incr)
        converter, augmentation, piler = build_augmentation()
        new_data: Dataset = concatenate_datasets([dataset["train"]] * aug_incr)
        dataset_train = new_data.map(lambda e: {'image':apply_augmentation(e['image'], augmentation, converter, piler), 'text': e['text']})
        dataset["train"] = dataset_train.shuffle()
    dataset = dataset.map(lambda e: {
    "pixel_values": processor(e['image'], return_tensors="pt").pixel_values.squeeze(),
    "labels": tokenizer(e['text'], return_tensors="pt", padding="max_length").input_ids.squeeze()})
    dataset = dataset.remove_columns(["image", "text"])
    dataset = dataset.with_format("torch")
    logging.info("Dataset loaded and processed")
    return dataset


def build_augmentation(
        random_erasing: bool = True,
        random_rotation: bool = True,
) -> Tuple[Callable, Dict[int, Callable]]:
    augment: Dict[int, Callable] = dict()
    p_rand_err: float = 0.5 if random_erasing else 0
    p_rand_rot: int = 10 if random_rotation else 0
    augment_0: Callable = torch.nn.Sequential(
        transforms.RandomAffine(degrees=p_rand_rot),
        transforms.GaussianBlur(21),
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.RandomAdjustSharpness(0.5),
        transforms.RandomErasing(p=p_rand_err),
    )
    augment[0] = torch.jit.script(augment_0)

    # Blurring
    augment_1: Callable = torch.nn.Sequential(
        transforms.RandomAffine(degrees=p_rand_rot),
        transforms.GaussianBlur(21),
        transforms.RandomAdjustSharpness(0.5),
        transforms.RandomErasing(p=p_rand_err),
    )
    augment[1] = torch.jit.script(augment_1)

    # Color Adjustments
    augment_2: Callable = torch.nn.Sequential(
        transforms.RandomAffine(degrees=p_rand_rot),
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.RandomAdjustSharpness(0.5),
        transforms.RandomErasing(p=p_rand_err),
    )
    augment[2] = torch.jit.script(augment_2)
    augment_3: Callable = torch.nn.Sequential(
        transforms.RandomAffine(degrees=p_rand_rot),
        transforms.RandomErasing(p=p_rand_err),
    )
    augment[3] = torch.jit.script(augment_3)
    augment[4] = lambda e: e
    converter: Callable = transforms.ToTensor()
    piler: Callable = transforms.ToPILImage()
    return converter, augment, piler 

def apply_augmentation(img: Any, augment: Dict[int, Callable], converter: Callable, piler: Callable) -> Any:
    choice: int = np.random.choice(range(len(augment.keys())))
    return piler(augment[choice](converter(img)))


class CERMetric:
    """CER-Class.

    Class to Compute Character-Error-Rate and keep necessary
    properties for efficient computing.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        """Instantiate Class.

        Compute Character-Error-Rate.

        :param tokenizer: Tokenizer, to decode predictions and labels.
        """
        self.tokenizer: PreTrainedTokenizerFast = tokenizer
        self.metric: CharErrorRate = CharErrorRate()

    def __call__(self, pred: torch.Tensor) -> typing.Dict[str, float]:
        """Compute Character Error Rate.

        :param pred: Predictions
        :return: Dict of {'cer': <computed cer value>}
        """
        labels_ids: torch.Tensor = pred.label_ids
        pred_ids: torch.Tensor = pred.predictions

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        cer: float = self.metric(pred_str, label_str)

        return {"cer": cer}
