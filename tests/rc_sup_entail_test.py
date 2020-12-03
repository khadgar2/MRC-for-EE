from argparse import Namespace
from base.utils import set_random_seed
from base.model import RCEntailModel
import argparse
import pytorch_lightning as pl
import itertools
import os
import torch
import shutil
from base.data.utils import make_few_shots
import json
from base.exp import conduct_experiments

params = {
    'query_type': [
        "IdentityQuery",
    ],
    'event_label_processor': [
        'IdentityEventProcessor',
    ],
    'seed': [0,1,2],
    'nsamps': ['all'],
    'use_event_description': ['none'],
    'nepochs': [5],
    'batch_size': [32]
}

gpus=[6]

base_path = 'exps/rc_model_sup/entail_only_event'
if __name__ == '__main__':
    conduct_experiments(
        'RCEntailModel',
        base_path,
        params,
        model_kwargs={
            'reverse_query': True,
            'accumulate_grad_batches': 12,
            'pretrain_model': 'bert-base-uncased',
        },
        gpus=gpus,
    )

