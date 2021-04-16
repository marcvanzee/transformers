#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Flax Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
import sys
from typing import Tuple, Union

import datasets
from dataclasses import dataclass, field
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from typing import Optional, List
from functools import partial

import transformers
from transformers import (
    BertConfig,
    FlaxBertForSequenceClassification,
    BertTokenizerFast,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    TensorType
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import logging as hf_logging

from flax import optim, jax_utils, struct
import jax
import jax.numpy as jnp
from flax.training.common_utils import shard, onehot
from flax import linen as nn


class Glue(struct.PyTreeNode):
  sentences: Tuple[str, Union[str, None]] = struct.field(pytree_node=False)
  num_labels: int = struct.field(pytree_node=False)
  is_regression: bool = struct.field(pytree_node=False)

glue_tasks = {
    "cola": Glue(("sentence", None), 2, False),
    "mnli": Glue(("premise", "hypothesis"), 3, False),
    "mnli-mm": Glue(("premise", "hypothesis"), 3, False),
    "mrpc": Glue(("sentence1", "sentence2"), 2, False),
    "qnli": Glue(("question", "sentence"), 2, False),
    "qqp": Glue(("question1", "question2"), 2, False),
    "rte": Glue(("sentence1", "sentence2"), 2, False),
    "sst2": Glue(("sentence", None), 2, False),
    "stsb": Glue(("sentence1", "sentence2"), 1, True),
    "wnli": Glue(("sentence1", "sentence2"), 2, False),
}

@dataclass
class FlaxTrainingArguments(TrainingArguments):
    """We use this to override some defaults in TrainingArguments."""
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=True, metadata={"help": "Whether to run predictions on the test set."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(
        metadata={
            "help": "The name of the task to train on: " + ", ".join(glue_tasks.keys())
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()
        

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

def rename_mnli_task(task_name, split):
    # The MNLI task consists of two tasks: MNLI-matched and MNLI-mismatched,
    # which we identify with resp. `mnli` and `mnli-mm` in this script.
    # Both tasks have the same training set, but a different validation and test
    # set. In the `datasets` library, the MNLI-matched validation and test sets
    # are in `mnli_matched`, and those of MNLI-mismatched are in
    # `mnli_mismatched`. Therefore we perform the following rewrites to make
    # sure we obtain the correct train, validation and test sets:
    #        
    # `task_name` | `datasets` train set | `dataesets` validation/test set
    # ------------|----------------------|--------------------------------
    # mnli        | mnli                 | mnli_matched
    # mnli-mm     | mnli                 | mnli_mismatched
    if not task_name.startswith("mnli"):
        return task_name
    if split == "train":
        return "mnli"
    elif task_name == "mnli":
        return "mnli_matched"
    else:
        return "mnli_mismatched"


def get_dataset(task_name, split, cache_dir, tokenizer, max_seq_length):
    dataset = load_dataset("glue", rename_mnli_task(task_name, split), split=split, cache_dir=cache_dir)

    sentence1, sentence2 = glue_tasks[task_name].sentences
    def tokenize_fn(examples):
        args = (examples[sentence1],)
        args += () if sentence2 is None else (sentence2,)
        return tokenizer(*args, max_length=max_seq_length, padding="max_length", truncation=True)

    cols_to_remove = dataset.column_names
    cols_to_remove.remove("label")  # Keep the label.
    dataset = dataset.map(tokenize_fn, remove_columns=cols_to_remove, batched=True)
    return dataset


def create_optimizer(model, learning_rate):

    # def adam_optimizer(weight_decay):
    #     return optim.Adam(learning_rate=learning_rate, beta1=0.9,
    #         beta2=0.999, eps=1e-6, weight_decay=weight_decay)

    # optimizer_decay_def = adam_optimizer(weight_decay=0.01)
    # optimizer_no_decay_def = adam_optimizer(weight_decay=0.0)
    # decay = optim.ModelParamTraversal(lambda path, _: 'bias' not in path)
    # no_decay = optim.ModelParamTraversal(lambda path, _: 'bias' in path)
    # optimizer_def = optim.MultiOptimizer(
    #   (decay, optimizer_decay_def), (no_decay, optimizer_no_decay_def))
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    optimizer = optimizer_def.create(model)
    return optimizer


def regression_loss_fn(logits, labels):
    return jnp.mean((logits[..., 0] - labels) ** 2)


def classification_loss_fn(logits, labels):
    logits = nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(onehot(labels, glue_task.num_labels) * logits, axis=-1))


def train_step(optimizer, batch, dropout_rng, apply_fn, glue_loss_fn, lr_scheduler_fn):
    print("Compiling train")
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
    targets = batch.pop("label")

    def loss_fn(params):
        logits = apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=False)[0]
        loss = glue_loss_fn(logits, targets)
        return loss, logits
    
    step = optimizer.state.step
    lr = lr_scheduler_fn(step)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grad = grad_fn(optimizer.target)
    grad = jax.lax.pmean(grad, "batch")
    # clip_grad_norm = 1.0
    # if clip_grad_norm is not None:
    #   grad_norm = sum([jnp.sum(x ** 2) for x in jax.tree_leaves(grad)])
    #   metrics['grad_norm'] = grad_norm
    #   grad_scale = jnp.where(
    #       grad_norm < clip_grad_norm, 1.0, clip_grad_norm / grad_norm)
    #   grad = jax.tree_map(lambda x: x * grad_scale, grad)
    optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
    metrics = {'loss': loss}
    return optimizer, metrics, new_dropout_rng


def create_learning_rate_scheduler(
        factors='constant * linear_warmup * rsqrt_decay',
        base_learning_rate=0.5,
        warmup_steps=1000,
        decay_factor=0.5,
        steps_per_decay=20000,
        steps_per_cycle=100000):
    """Creates learning rate schedule.
    Interprets factors in the factors string which can consist of:
    * constant: interpreted as the constant value,
    * linear_warmup: interpreted as linear warmup until warmup_steps,
    * rsqrt_decay: divide by square root of max(step, warmup_steps)
    * decay_every: Every k steps decay the learning rate by decay_factor.
    * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.
    * linear_decay: Linear decay, uses steps_per_cycle parameter.
    Args:
      factors: a string with factors separated by '*' that defines the schedule.
      base_learning_rate: float, the starting constant for the lr schedule.
      warmup_steps: how many steps to warm up for in the warmup schedule.
      decay_factor: The amount to decay the learning rate by.
      steps_per_decay: How often to decay the learning rate.
      steps_per_cycle: Steps per cycle when using linear/cosine decay.
    Returns:
      a function learning_rate(step): float -> {'learning_rate': float}, the
      step-dependent lr.
    """
    factors = [n.strip() for n in factors.split('*')]

    def step_fn(step):
        """Step to learning rate function."""
        ret = 1.0
        for name in factors:
            if name == 'constant':
                ret *= base_learning_rate
            elif name == 'linear_warmup':
                ret *= jnp.minimum(1.0, step / warmup_steps)
            elif name == 'rsqrt_decay':
                ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
            elif name == 'rsqrt_normalized_decay':
                ret *= jnp.sqrt(warmup_steps)
                ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
            elif name == 'decay_every':
                ret *= (decay_factor**(step // steps_per_decay))
            elif name == 'cosine_decay':
                progress = jnp.maximum(0.0,
                                      (step - warmup_steps) / float(steps_per_cycle))
                ret *= jnp.maximum(0.0,
                                   0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
            elif name == 'linear_decay':
                progress = jnp.maximum(0.0,
                                      (step - warmup_steps) / float(steps_per_cycle))
                ret *= 1.0 - progress
            else:
                raise ValueError('Unknown factor %s.' % name)
        return jnp.asarray(ret, dtype=jnp.float32)

    return step_fn


def get_batches(rng, train_dataset, batch_size):
  train_ds_size = len(train_dataset)
  steps_per_epoch = train_ds_size // batch_size
  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  for perm in perms:
    batch = train_dataset[perm]
    batch = {k: jnp.array(v) for k, v in batch.items() }
    yield shard(batch)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FlaxTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        hf_logging.set_verbosity_info()
        hf_logging.enable_explicit_format()
        hf_logging.enable_default_handler()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    config = BertConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=glue_tasks[data_args.task_name].num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = BertTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = FlaxBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        seed=training_args.seed,
    )

    is_regression = glue_tasks[data_args.task_name].is_regression
    num_labels = glue_tasks[data_args.task_name].num_labels

    get_split = partial(get_dataset, task_name=data_args.task_name, cache_dir=model_args.cache_dir, 
                        tokenizer=tokenizer, max_seq_length=data_args.max_seq_length)

    train_dataset = get_split(split="train") if training_args.do_train else None
    eval_dataset = get_split(split="validation") if training_args.do_eval else None
    test_dataset = get_split(split="test") if training_args.do_predict else None

    optimizer = create_optimizer(model.params, training_args.learning_rate)
    optimizer = jax_utils.replicate(optimizer)

    train_ds_size = len(train_dataset)
    train_batch_size = training_args.train_batch_size
    num_epochs = int(training_args.num_train_epochs)
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_epochs

    learning_rate_fn = create_learning_rate_scheduler(
        factors='constant * linear_warmup * linear_decay',
        base_learning_rate=training_args.learning_rate,
        warmup_steps=max(training_args.warmup_steps, 1),
        steps_per_cycle=num_train_steps - training_args.warmup_steps,
    )

    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())
    p_train_step = jax.pmap(
            partial(
                train_step, 
                apply_fn=model.__call__,
                glue_task=glue_tasks[data_args.task_name],
                lr_scheduler_fn=learning_rate_fn),
            axis_name="batch",
            donate_argnums=(0,))

    if training_args.do_train:  
        i = 0  
        for epoch in range(1, num_epochs + 1):
            rng, input_rng = jax.random.split(rng)
            for batch in get_batches(input_rng, train_dataset, train_batch_size):
                optimizer, metrics, dropout_rngs = p_train_step(optimizer, batch, dropout_rngs)
                # logging.info('metrics: %s', metrics)
                if i % 10 == 0:
                  print(f"step {i}: {metrics}")
                i += 1
    
    if training_args.do_eval:
        for batch in get_batches(input_rng, train_dataset, train_batch_size):
                optimizer, metrics, dropout_rngs = p_train_step(optimizer, batch, dropout_rngs)
                # logging.info('metrics: %s', metrics)
                if i % 10 == 0:
                  print(f"step {i}: {metrics}")
                i += 1



if __name__ == "__main__":
    main()
