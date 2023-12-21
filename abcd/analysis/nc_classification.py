import os
import pandas as pd
from collections import OrderedDict
import torch
torch.manual_seed(0)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
from abcd.data.read_data import get_subjects_events_sf, subject_cols_to_events, add_event_vars
import abcd.data.VARS as VARS
from abcd.data.define_splits import SITES, save_restore_sex_fmri_splits
from abcd.data.divide_with_splits import divide_events_by_splits
from abcd.data.var_tailoring.normalization import normalize_var
from abcd.data.pytorch.get_dataset import PandasDataset
from abcd.training.ClassifierTrainer import ClassifierTrainer
from abcd.local.paths import core_path, output_path
import abcd.data.VARS as VARS
from abcd.exp.Experiment import Experiment
import abcd.utils.io as io
import importlib
import sys
from matplotlib import pyplot as plt

# %%
config = {'target_col': 'nihtbx_fluidcomp_fc',
          'features': ['fmri', 'smri'],
          'model': ['abcd.models.classification.FullyConnected', 'FullyConnected3'],
          'lr': 1e-3,
          'batch_size': 64,
          'nr_epochs': 150}

print("Print statements are live and working")
breakpoint()
#todo: make new experiment when all data works
exp = Experiment(name='neurocog_classification_debugxiv', config=config)
print("Reached here")

# %%
# Fetch subjects and events
subjects_df, events_df = get_subjects_events_sf()
print("There are {} subjects and {} visits with imaging".format(len(subjects_df), len(events_df)))

# %%
events_df

# %%
subjects_df

# %%
# Add the target to the events df, if not there
target_col = config['target_col']
if target_col not in events_df.columns:
    events_df = add_event_vars(events_df, "/Users/brentju/Desktop/abcd/nc_y_nihtb.csv", [target_col])
    
    
# Add scanner data to table if not exists
scanner_cols = ['mri_info_manufacturer', 'mri_info_manufacturersmn']
for col in scanner_cols:
    if col not in events_df.columns:
        mri_df = io.load_df("/Users/brentju/Desktop/abcd/mri_y_adm_info.csv", sep =',', cols=["src_subject_id", "eventname"]+scanner_cols)
        events_df = pd.merge(events_df, mri_df, on=["src_subject_id", "eventname"])  
        break


# Add data from family, environment
family_income_col = ['demo_comb_income_v2']
# Conflict Subscale from the Family Environment Scale Sum of Youth Report
environment_cols = ['fes_y_ss_fc']

income_df = io.load_df(os.path.join(core_path, 'abcd-general', 'abcd_p_demo.csv'), sep =',', cols=["src_subject_id", "eventname"]+family_income_col)
env_df = io.load_df(os.path.join(core_path, 'culture-environment', 'ce_y_fes.csv'), sep =',', cols=["src_subject_id", "eventname"]+environment_cols)

events_df = pd.merge(events_df, env_df, on=["src_subject_id", "eventname"])  
events_df = pd.merge(events_df, income_df, on=["src_subject_id", "eventname"]) 

events_df = events_df.dropna()
events_df

# %%
def discretize_column_inplace(events_df, column_name):
    _, bins = pd.qcut(events_df[column_name], q=[0, 0.25, 0.5, 0.75, 1], labels=False, retbins=True, duplicates='drop')
    events_df[column_name] = pd.cut(events_df[column_name], bins=bins, labels=False, include_lowest=True)
    return events_df

events_df = discretize_column_inplace(events_df, target_col)
set(events_df[target_col])

# %%
siemens_table = events_df[events_df['mri_info_manufacturer'] == "SIEMENS"]
ge_table = events_df[events_df['mri_info_manufacturer'] == "GE MEDICAL SYSTEMS"]
philips_table = events_df[events_df['mri_info_manufacturer'] == "Philips Medical Systems"]
msg = f'{len(siemens_table)} entries for Siemens, {len(ge_table)} entries for GE Medical, and {len(philips_table)} for Philips'
print(msg)

# %%
# Define features
features_fmri = list(VARS.NAMED_CONNECTIONS.keys())
features_smri = [var_name + '_' + parcel for var_name in VARS.DESIKAN_STRUCT_FEATURES.keys() for parcel in VARS.DESIKAN_PARCELS[var_name] + VARS.DESIKAN_MEANS]
feature_cols = ['demo_comb_income_v2','fes_y_ss_fc']
if 'fmri' in config['features']:
    feature_cols += features_fmri
if 'smri' in config['features']:
    feature_cols += features_smri

# %%
# Normalize features
for var_id in feature_cols:
    events_df = normalize_var(events_df, var_id, var_id)

# %%
siemens_train = siemens_table[:int(0.8*len(siemens_table))]
ge_train = ge_table[:int(0.8*len(ge_table))]
philips_train = ge_table[:int(0.8*len(philips_table))]


siemens_test = siemens_table[int(0.8*len(siemens_table)):]
ge_test = ge_table[int(0.8*len(ge_table)):]
philips_test = ge_table[int(0.8*len(philips_table)):]

# %%
# Define PyTorch datasets and dataloaders
s_datasets = OrderedDict([('SiemensTrain', PandasDataset(siemens_train, feature_cols, target_col)),
            ('SiemensTest', PandasDataset(siemens_test, feature_cols, target_col))])

g_datasets = OrderedDict([('GETrain', PandasDataset(ge_train, feature_cols, target_col)),
            ('GETest', PandasDataset(ge_test, feature_cols, target_col))])

p_datasets = OrderedDict([('PhilipsTrain', PandasDataset(philips_train, feature_cols, target_col)),
            ('PhilipsTest', PandasDataset(philips_test, feature_cols, target_col))])

# %%
# Create dataloaders
batch_size = config['batch_size']
s_dataloaders = OrderedDict([(dataset_name, DataLoader(dataset, batch_size=batch_size, shuffle=True))
    for dataset_name, dataset in s_datasets.items()])

g_dataloaders = OrderedDict([(dataset_name, DataLoader(dataset, batch_size=batch_size, shuffle=True))
    for dataset_name, dataset in g_datasets.items()])

p_dataloaders = OrderedDict([(dataset_name, DataLoader(dataset, batch_size=batch_size, shuffle=True))
    for dataset_name, dataset in p_datasets.items()])

# %%
device = "cpu"
print(device)

# %%
# Define model
labels = [target_col]
models_path = os.path.join(exp.path, 'models')
module = importlib.import_module(config['model'][0])
model = getattr(module, config['model'][1])(save_path=models_path, labels=labels, input_size=len(feature_cols))
#model = FullyConnected5(save_path=models_path, labels=labels, input_size=len(feature_cols))
model = model.to(device)

# %%
# Define optimizer and trainer
learning_rate = config['lr']
loss_f = nn.CrossEntropyLoss()
trainer_path = os.path.join(exp.path, 'trainer')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
trainer = ClassifierTrainer(trainer_path, device, optimizer, loss_f, labels=labels)
s_dataloaders

# %% [markdown]
# # Methodology:
# * Train the model on the Siemens train set. Evaluate on test set of all 3 models
# * Continue training of the model on the GE and Philips datasets as well, evaluating on all 3 datasets as well.

# %%
nr_epochs = config['nr_epochs']
print("Entering breakpoint")
trainer.train(model, s_dataloaders['SiemensTrain'], s_dataloaders, 
              nr_epochs=nr_epochs, starting_from_epoch=0,
              print_loss_every=int(nr_epochs/10), eval_every=int(nr_epochs/10), export_every=int(nr_epochs/5), verbose=True)


