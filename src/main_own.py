"""Main File.

Machine Learning
Tsinghua University 
Winter Term 23/24
"""

import logging
import os
import pathlib
import wandb

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tokenizers.pre_tokenizers import Whitespace 
from transformers import (AutoTokenizer, TrOCRProcessor,
                          Seq2SeqTrainer, GPT2Config,
                          AutoImageProcessor,
                          SwinConfig, SwinModel,
                          Seq2SeqTrainingArguments,
                          VisionEncoderDecoderConfig,
                          PreTrainedTokenizerFast,
                          VisionEncoderDecoderModel,
                          default_data_collator)
import functools
from tokenizers import Tokenizer, decoders, processors
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from datasets import DatasetDict
from utils import CERMetric, get_dataset
from torch.utils.data import Dataset as torch_dataset
from datasets import load_dataset




def train_tokenizer(dataset_identifier, batch_size: int=1000, tokenizer_path: str='./tokenizer'):
    dataset: DatasetDict = load_dataset(dataset_identifier)
    text = dataset['train']['text'] + dataset['validation']['text'] + dataset['test']['text']
    alphabet: set = set(functools.reduce(lambda x, y: x + y, text, ''))

    special_tokens_list: list = ["[BOS]", "[UNK]", "[EOS]"]

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=len(alphabet) + 10, special_tokens=special_tokens_list
    )
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.post_processor = TemplateProcessing(
        single="$A [EOS]",
        special_tokens=[("[EOS]", 2)],
    )


    def batch_iterator():
        for i in range(0, len(dataset), batch_size):
            yield text[i : i + batch_size]

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save(tokenizer_path)



@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """TODO

    :param cfg: Configuration for Hydra. DO NOT PASS ARGUMENT!
    """
    # Technical Setup
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["MLFLOW_FLATTEN_PARAMS"] = "true"
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(cfg.seed)
    logging.info("Seeded")
    logging.info('Using config: \n%s', OmegaConf.to_yaml(cfg))
    wandb.init(
        **cfg.wandb_args
    )

    config_encoder = SwinConfig()
    config_decoder = GPT2Config()
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        config_encoder, config_decoder
    )
    model = VisionEncoderDecoderModel(config=config)
    processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    train_tokenizer("agomberto/FrenchCensus-handwritten-texts")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='./tokenizer')


    # TODO: To config
    dataset: DatasetDict = get_dataset("agomberto/FrenchCensus-handwritten-texts", processor, tokenizer, cfg.augment.augment, cfg.augment.aug_incr, cfg.debug)
    dataset_train: torch_dataset = dataset["train"].with_format("torch")
    dataset_test: torch_dataset = dataset["test"].with_format("torch")

    tokenizer.model_max_length = cfg.nlg_configs.max_length

    # Set up tokenizer and models for task
    special_tokens_dict = {
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]",
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # # Set Beam-Search Params
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = cfg.nlg_configs.max_length
    model.config.early_stopping = cfg.nlg_configs.early_stopping
    model.config.no_repeat_ngram_size = cfg.nlg_configs.no_repeat_ngram_size
    model.config.length_penalty = cfg.nlg_configs.length_penalty
    model.config.num_beams = cfg.nlg_configs.num_beams

    training_args: Seq2SeqTrainingArguments = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        num_train_epochs=cfg.training_configs.epochs,
        per_device_train_batch_size=cfg.training_configs.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training_configs.per_device_eval_batch_size,
        fp16=torch.cuda.is_available(),
        output_dir="./",
        logging_steps=cfg.training_configs.logging_steps,
        save_steps=cfg.training_configs.save_steps,
        eval_steps=cfg.training_configs.eval_steps,
        report_to=cfg.training_configs.report_to,
        run_name=cfg.training_configs.run_name,
    )

    cer_fun: CERMetric = CERMetric(tokenizer)

    trainer: Seq2SeqTrainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor,
        args=training_args,
        compute_metrics=cer_fun,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        data_collator=default_data_collator,
    )
    trainer.train()
    metrics: dict = trainer.evaluate()
    logging.info(metrics)
    wandb.log(metrics)

    # Save Model
    pathlib.Path(cfg.training_configs.target_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.training_configs.target_path)
    logging.info(f"Saved model at: {cfg.training_configs.target_path}")

    wandb.finish()

if __name__ == "__main__":
    main()
