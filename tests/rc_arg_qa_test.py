import os
from argparse import Namespace
from base.utils import set_random_seed
from base.model import RCArgumentModel
import argparse
import pytorch_lightning as pl
import itertools
import torch
import shutil
import json

torch.multiprocessing.set_sharing_strategy('file_system')
gpus=[0,1, 2,3,6,7]
agb=1
batch_size=50
# set_random_seed(default_args['seed'])
params = {
    'arg_template': ['ARG_EQ2'],
    'fs_type': ['argument'],
    'nsamps': [1,3,5,7,9,'all'],
    'seed': [0,1,2]
}
#params['nsamps'].reverse()

def zip_all_params(all_params):
    return [{p: q for p, q in zip(all_params.keys(), x)} for x in list(itertools.product(*list(all_params.values())))]


zipped_params = zip_all_params(params)

base_path = os.path.join('exps', 'rc_arg_model_large', 'few_shot_test')

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

exp_name_set = set()
for iarg, args in enumerate(zipped_params):
    print("Experiment {}/{}".format(iarg, len(zipped_params)))
    exp_name = ''
    arg_template = args['arg_template']
    fs_type = args['fs_type']
    nsamps = args['nsamps']
    seed = args['seed']
    if args['nsamps'] != 'all':
        exp_name_set.add(tuple(args.values()))
    else:
        if (arg_template, 'all', seed) in exp_name_set:
            continue
        exp_name_set.add((arg_template, 'all', seed))
    '''
    if arg_template in ('ARG_EQ_TRI_1', 'ARG_EQ_TRI_2'):
        if (arg_template, seed) in exp_name_set:
            continue
        nsamps = 'all'
        exp_name_set.add((arg_template, seed))
    '''
    if arg_template == 'ARG_DESC_WITHOUT_EN':
        desc_type = 'SQ'
    elif arg_template == 'ARG_DESC_WITH_EN':
        desc_type = 'PQ'
    else:
        desc_type = None
    for x, y in args.items():
        exp_name += '{}={}-'.format(x, y)
    if len(exp_name) > 0:
        exp_name = exp_name[:-1]
    print(exp_name)
    pretrained_model = 'deepset/bert-large-uncased-whole-word-masking-squad2'  #if arg_template in ('ARG_DESC_WITHOUT_EN', 'ARG_EQ_TRI_2') else 'bert-large-uncased'
    exp_path = os.path.join(base_path, exp_name)
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

    model = RCArgumentModel(Namespace(
        input_train_processed='processed_arg_data/train_samp={}_fs-type={}_desc_type={}.pickle'.format(args['nsamps'], args['fs_type'], desc_type),
        input_test_processed="processed_arg_data/test_fs-type={}_desc_type={}.pickle".format(args['fs_type'], desc_type),
        input_val_processed="processed_arg_data/dev_fs-type={}_desc_type={}.pickle".format(args['fs_type'], desc_type),
        batch_size=batch_size,
        nepochs=7,
        accumulate_grad_batches=agb,
        desc_type=desc_type,
        nsamp=args['nsamps'],
        query_template=arg_template,
        reverse_query=False,
        pretrain_model=pretrained_model,
        val_pred_path='res/event_preds/dev_preds.npy',
        test_pred_path='res/event_preds/test_preds.npy'
    )
    )

    trainer = pl.Trainer(
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            filepath=exp_path,
            verbose=True,
            #monitor='arg_f1',
            monitor='ev_pred_arg_f1',
            mode='max',
            save_top_k=1,
        ),
        gpus=gpus,
        max_epochs=7,
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
