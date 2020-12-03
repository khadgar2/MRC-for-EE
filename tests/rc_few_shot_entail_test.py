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

torch.multiprocessing.set_sharing_strategy('file_system')

params = {
    'query_type': [
        "ST1",
    ],
    'event_label_processor': [
        'IdentityEventProcessor',
    ],
    'seed': [0,1,2,3,4],
    'nsamps': [1,3,5,7,9],
    'use_event_description': ['none', 'postfix'],
    'neutral_as': ['negative'],
    'nepochs': [5],
    'max_desc_sentences':[1, 5],
}

def zip_all_params(all_params):
    return [{p: q for p, q in zip(all_params.keys(), x)} for x in list(itertools.product(*list(all_params.values())))]

zipped_params = zip_all_params(params)

base_path = 'exps/rc_model_desc/few_shot_entail'
try:
    os.makedirs(base_path)
except FileExistsError:
    pass

with open(os.path.join(base_path, 'params.json'), 'w') as f:
    json.dump(params, f)


results_path = os.path.join(base_path, 'results.csv')

if not os.path.exists(results_path):
    first_exp = True
else:
    with open(results_path, 'r') as f:
        if len(f.readlines()) == 1:
            first_exp = True
        else:
            first_exp = False

if first_exp:
    with open(results_path, 'w') as f:
        f.write(','.join(list(params.keys())))
        f.write(",")

nexps = len(zipped_params)
for iarg, args in enumerate(zipped_params):
    exp_name = ''
    for x, y in args.items():
        exp_name += '{}:{}-'.format(x, y)
    if len(exp_name) > 0:
        exp_name = exp_name[:-1]

    exp_path = os.path.join(base_path, exp_name)
    print("Experiment {}/{}".format(
        iarg+1, nexps
    ))
    print("Directory: ")
    print(exp_path)
    try:
        os.makedirs(exp_path)
    except FileExistsError:
        if os.path.exists(os.path.join(exp_path, "FINISHED")):
            print("{} directory exists and has FINISHED flag. Will skip it. ".format(exp_name))
            continue
        else:
            print("{} directory exists but unfinished. Restarting it.".format(exp_name))
            shutil.rmtree(exp_path)
            os.makedirs(exp_path)

    pl.seed_everything(args['seed'])

    model = RCEntailModel(Namespace(
            input_train="ACE05/train_process.json",
            input_test="ACE05/test_process.json",
            input_dev="ACE05/dev_process.json",
            batch_size=32*2,
            nepochs=args['nepochs'],
            accumulate_grad_batches=1,
            event_query_template=args['query_type'],
            event_label_processor=args['event_label_processor'],
            use_event_description=args['use_event_description'],
            max_desc_sentences=args['max_desc_sentences'],
            reverse_query=True,
            nsamps=args['nsamps'],
            #pretrain_model='ishan/distilbert-base-uncased-mnli',
            pretrain_model='textattack/bert-base-uncased-MNLI',
            neutral_as=args['neutral_as']
        )
    )

    trainer = pl.Trainer(
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            filepath=exp_path,
            verbose=True,
            monitor='ev_f1',
            mode='max',
            save_top_k=1,
        ),
        gpus=[3,4,5],
        max_epochs=args['nepochs'],
        distributed_backend='dp',
    )
    trainer.fit(model)
    trainer.test()

    # if first, write callback metrics column names
    with open(results_path, 'a') as f:
        if first_exp:
            first_exp = False
            keys = trainer.callback_metrics.keys()
            f.write(",".join(keys))
            f.write("\n")

        args_str = ",".join([str(x) for x in args.values()])
        results = {x: float(t) if isinstance(t, torch.Tensor) else t 
                    for x, t in trainer.callback_metrics.items()}
        res_str = ",".join([str(x) for x in results.values()])
        f.write(args_str + "," + res_str + "\n")
    with open(os.path.join(exp_path, "FINISHED"), 'w+') as f:
        f.write("")

