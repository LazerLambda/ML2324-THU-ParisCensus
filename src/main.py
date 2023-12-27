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
from transformers import (AutoTokenizer, TrOCRProcessor,
                          Seq2SeqTrainer,
                          Seq2SeqTrainingArguments,
                          VisionEncoderDecoderModel,
                          default_data_collator)
from datasets import DatasetDict
from utils import CERMetric, get_dataset
from torch.utils.data import Dataset as torch_dataset


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

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") 
    model = VisionEncoderDecoderModel.from_pretrained("agomberto/trocr-base-printed-fr")
    tokenizer = AutoTokenizer.from_pretrained("agomberto/trocr-base-printed-fr")


    # TODO: To config
    dataset: DatasetDict = get_dataset("agomberto/FrenchCensus-handwritten-texts", processor, tokenizer, cfg.augment.augment, cfg.augment.aug_incr, cfg.debug)
    dataset_train: torch_dataset = dataset["train"].with_format("torch")
    dataset_test: torch_dataset = dataset["test"].with_format("torch")

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
