import pandas as pd

pd.options.mode.chained_assignment = None
import numpy as np
import torch

dtype = torch.cuda.FloatTensor
from torch import nn
import numpy as np
import random
from tqdm import tqdm
import polygnn_trainer as pt
from skopt import gp_minimize
from sklearn.model_selection import train_test_split
from os import mkdir
import time
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from convert2dict import convert2dict
from os.path import exists

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
args = parser.parse_args()

# #################
# input variables 
# #################

# Columns in your dataset to ignore
removeCols = ['ID']

# Columns in your dataset that should be passed into the physics equations
physics_cols = ['temperature']

# Name of the folder for results to be stored
outside_folder = 'Physics_Example/'

# The columns to be stored in the prediction output files
# This set of final_cols ignores ID and fp_ columns
final_cols = ['smiles', 'log_MW', 'molality','temperature', 'prop', 'value']

# True if you want to pre-split your data, False otherwise
self_split = True

# If self_split = True, put None here
# If self_split = False, put the filename of the relevant dataset here
data_filename = None

# If self_split = True, list the folder name where the predefined splits are located
# if self_split = False, put None here
splits_folder = './Example_Data/' 

# If multiple models are to be trained, this list contains folder names the respective data is held
# In this file, example1 is a single task model and example 2 is a multi task model
folders = ['example1', 'example2'] 


# The keys of PROPERTY_GROUPS should be the values of folders
# The values of PROPERTY_GROUPS should be the property names relevant for a specific example
PROPERTY_GROUPS = {
    'example1': ['property1'],
    'example2': ['property1', 'property2']
}

'''
# This is a way that I automated it for many simultaneous runs
for folder in folders:
    PROPERTY_GROUPS[folder] = target_properties
'''
# #########
# constants
# #########

# Optional Constants for fast training

RANDOM_SEED = 100
HP_EPOCHS = 20  # companion paper used 200
SUBMODEL_EPOCHS = 100  # companion paper used 1000
N_FOLDS = 3  # companion paper used 5
HP_NCALLS = 10  # companion paper used 25
MAX_BATCH_SIZE = 50  # companion paper used 450
capacity_ls = list(range(2, 3))
weight_decay = 0

# Constants for good training
isMLP = True
'''
RANDOM_SEED = 100
HP_EPOCHS = 200
SUBMODEL_EPOCHS = 1000
N_FOLDS = 5
HP_NCALLS = 25
MAX_BATCH_SIZE = 450
capacity_ls = list(range(2, 7))
weight_decay = 0
'''
N_FEATURES = 512
OPT_CAPACITY = 2  # optimal capacity

# #########
start = time.time()

# fix random seeds
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Choose the device to train our models on.
if args.device == "cpu":
    device = "cpu"
elif args.device == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # specify GPU


def morgan_featurizer(smile):
    smile = smile.replace("*", "H")
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=N_FEATURES, useChirality=True
    )
    fp = np.expand_dims(fp, 0)
    return Data(x=torch.tensor(fp, dtype=torch.float))


# Make a directory to save our models in.
if not exists(f'./{outside_folder}'):
    mkdir(f'./{outside_folder}')

# Train one model per group. We only have one group, "electronic", in this
# example file.
num_run = -1
for group in PROPERTY_GROUPS:
    num_run += 1

    # Load data. This data set is a subset of the data used to train the
    # electronic-properties MT models shown in the companion paper. The full
    # data set can be found at khazana.gatech.edu.
    if self_split:
        trainfile = splits_folder + f'{group}/trainset.csv'
        testfile = splits_folder + f'{group}/testset.csv'
        raw_train_data = pd.read_csv(trainfile)
        raw_test_data = pd.read_csv(testfile)

        train_cols = raw_train_data.keys()
        test_cols = raw_test_data.keys()

        not_in_test = [col for col in train_cols if not(col in test_cols)]

        for col in not_in_test:
            raw_test_data[col] = 0

    else:
        raw_data = pd.read_csv(data_filename, index_col=0)
        raw_train_data, raw_test_data = train_test_split(
            raw_data,
            test_size=0.2,
            stratify=raw_data.prop,
            #random_state=RANDOM_SEED,
        )
    test_results = raw_test_data.copy()
    train_results = raw_train_data.copy()

    # The sample data does not contain any graph features.
    train_data, N_FEATURES = convert2dict(raw_train_data, removeCols, physics_cols, isMLP)
    test_data, N_FEATURES = convert2dict(raw_test_data, removeCols, physics_cols, isMLP)
    graph_train = train_data['graph_feats']
    graph_test = test_data['graph_feats']


    graph_train_keys = list(graph_train[0].keys())
    graph_test_keys = list(graph_test[0].keys())

    print('graph_train_keys')
    print(graph_train_keys)

    print('graph_test_keys')
    print(graph_test_keys)
    '''
    not_contained = [col for col in graph_train_keys if not (col in graph_test_keys)]

    for x in range(len(graph_test)):
        for col in not_contained:
            graph_test[x][col] = 0
    test_data['graph_feats'] = graph_test
    '''

    assert len(train_data['graph_feats'][0].keys()) == len(test_data['graph_feats'][0].keys())

    print(f'\n\nlen of train data: {len(train_data)}\n\n')


    assert len(train_data) > len(test_data)


    prop_cols = sorted(PROPERTY_GROUPS[group])
    print(
        f"Working on group {group}. The following properties will be modeled: {prop_cols}",
        flush=True,
    )

    nprops = len(prop_cols)
    if nprops == 1:
        selector_dim = 0
    else:
        selector_dim = nprops
    # Define a directory to save the models for this group of properties.
    root_dir = f'./{outside_folder}{group}'

    print(f'\n\nlen of train data: {len(train_data)}\n\n')
    group_train_data = train_data.loc[train_data.prop.isin(prop_cols), :]
    group_test_data = test_data.loc[test_data.prop.isin(prop_cols), :]
    print(f'\n\nlen of group train data before: {len(group_train_data)}\n\n')
    ######################
    # prepare data
    ######################
    #group_train_inds = group_train_data.index.values.tolist()
    group_train_data['traintest'] = 'train'
    #print(len(group_train_inds))
    #print(group_train_inds)
    #group_test_inds = group_test_data.index.values.tolist()
    group_test_data['traintest'] = 'test'
    group_data = pd.concat([group_train_data, group_test_data], ignore_index=False)
    #print(len(group_test_inds))
    #print(group_test_inds)

    if isMLP:
        smiles_feats = None
    else:
        smiles_feats = morgan_featurizer

    group_data, scaler_dict = pt.prepare.prepare_train(
        group_data, smiles_featurizer=smiles_feats, root_dir=root_dir
    )
    print([(k, str(v)) for k, v in scaler_dict.items()])
    group_train_data = group_data.loc[group_data['traintest'] == 'train', :]
    group_test_data = group_data.loc[group_data['traintest'] == 'test', :]
    print(f'\n\nlen of group train data after: {len(group_train_data)}\n\n')

    # ###############
    # do hparams opt
    # ###############
    # split train and val data
    group_fit_data, group_val_data = train_test_split(
        group_train_data,
        test_size=0.2,
        stratify=group_train_data.prop,
        random_state=RANDOM_SEED,
    )

    fit_pts = group_fit_data.data.values.tolist()
    val_pts = group_val_data.data.values.tolist()

    print(
        f"\nStarting hp opt. Using {len(fit_pts)} data points for fitting, {len(val_pts)} data points for validation."
    )
    # create objective function
    def obj_func(x):
        hps = pt.hyperparameters.HpConfig()
        hps.set_values(
            {
                "r_learn": 10 ** x[0],
                "batch_size": x[1],
                "dropout_pct": x[2],
                "capacity": OPT_CAPACITY,
                "activation": nn.functional.leaky_relu,
            }
        )
        print("Using hyperparameters:", hps)
        tc_search = pt.train.trainConfig(
            hps=hps,
            device=device,
            amp=False,  # False since we are on T2
            multi_head=False,
            loss_obj=pt.loss.sh_mse_loss(),
        )  # trainConfig for the hp search
        tc_search.epochs = HP_EPOCHS

        model = pt.models.PhysicsInformed(
            input_dim=N_FEATURES + selector_dim,
            output_dim=2,
            hps=hps,
        )
        val_rmse = pt.train.train_submodel(
            model,
            fit_pts,
            val_pts,
            scaler_dict,
            tc_search,
        )
        return val_rmse

    # create hyperparameter space
    hp_space = [
        (np.log10(0.0003), np.log10(0.03)),  # learning rate
        (round(0.25 * MAX_BATCH_SIZE), MAX_BATCH_SIZE),  # batch size
        (0, 0.5),  # dropout
    ]

    # obtain the optimal point in hp space
    opt_obj = gp_minimize(
        func=obj_func,  # defined offline
        dimensions=hp_space,
        n_calls=HP_NCALLS,
        random_state=RANDOM_SEED,
    )
    # create an HpConfig from the optimal point in hp space
    optimal_hps = pt.hyperparameters.HpConfig()
    optimal_hps.set_values(
        {
            "r_learn": 10 ** opt_obj.x[0],
            "batch_size": opt_obj.x[1],
            "dropout_pct": opt_obj.x[2],
            "capacity": OPT_CAPACITY,
            "activation": nn.functional.leaky_relu,
        }
    )
    print(f"Optimal hps are {opt_obj.x}")
    # clear memory
    del group_fit_data
    del group_val_data

    # ################
    # Train submodels
    # ################
    tc_ensemble = pt.train.trainConfig(
        amp=False,  # False since we are on T2
        loss_obj=pt.loss.sh_mse_loss(),
        hps=optimal_hps,
        device=device,
        multi_head=False,
    )  # trainConfig for the ensemble step
    tc_ensemble.epochs = SUBMODEL_EPOCHS
    print(f"\nTraining ensemble using {len(group_train_data)} data points.")
    pt.train.train_kfold_ensemble(
        dataframe=group_train_data,
        model_constructor=lambda: pt.models.PhysicsInformed(
            input_dim=N_FEATURES + selector_dim,
            output_dim=2,
            hps=optimal_hps,
        ),
        train_config=tc_ensemble,
        submodel_trainer=pt.train.train_submodel,
        augmented_featurizer=None,
        scaler_dict=scaler_dict,
        root_dir=root_dir,
        n_fold=N_FOLDS,
        random_seed=RANDOM_SEED,
    )
    ##########################################
    # Load and evaluate ensemble on test data
    ##########################################
    print("\nRunning predictions on test data", flush=True)
    ensemble = pt.load.load_ensemble(
        root_dir,
        pt.models.PhysicsInformed,
        device,
        {
            "input_dim": N_FEATURES + selector_dim,
            "output_dim": 2,
        },
    )
    # remake "group_test_data" so that "graph_feats" contains dicts not arrays
    group_test_data = test_data.loc[
        test_data.prop.isin(prop_cols),
        :,
    ]
    y_test, y_mean_hat_test, y_std_hat_test, lnA_test, Ea_test, _selectors_test = pt.infer.eval_ensemble(
        model=ensemble,
        root_dir=root_dir,
        dataframe=group_test_data,
        smiles_featurizer=smiles_feats,
        device=device,
        ensemble_kwargs_dict={"monte_carlo": False},
    )

    y_train, y_mean_hat_train, y_std_hat_train, lnA_train, Ea_train,  _selectors_train = pt.infer.eval_ensemble(
        model=ensemble,
        root_dir=root_dir,
        dataframe=group_train_data,
        smiles_featurizer=smiles_feats,
        device=device,
        ensemble_kwargs_dict={"monte_carlo": False},
    )

    print(train_results)
    print(test_results)
    print(len(y_train))
    print(len(y_test))
    print(len(group_train_data))

    test_results = test_results[final_cols]
    test_results['y'] = y_test
    test_results['y_mean_hat'] = y_mean_hat_test
    test_results['y_std_hat'] = y_std_hat_test
    test_results['lnA'] = lnA_test
    test_results['Ea'] = Ea_test

    train_results = train_results[final_cols]
    if len(prop_cols) == 1:
        train_results = train_results.loc[train_results['prop'] == 'exp_conductivity', :]
    elif len(prop_cols) != 2 and prop_cols == ['exp_conductivity']:
        train_results = train_results.loc[train_results['prop'] == 'exp_conductivity', :]
    train_results['y'] = y_train
    train_results['y_mean_hat'] = y_mean_hat_train
    train_results['y_std_hat'] = y_std_hat_train
    train_results['lnA'] = lnA_train
    train_results['Ea'] = Ea_train

    if not exists('./results/'):
        mkdir('./results/')

    folder = './results/' + outside_folder
    if not exists(folder):
        mkdir(folder)
    
    folder = folder + group
    if not exists(folder):
        mkdir(folder)
    
    testloc = folder + '/test_results.csv'
    trainloc = folder + '/train_results.csv'

    test_results.to_csv(testloc)
    train_results.to_csv(trainloc)

    pt.utils.mt_print_metrics(
        y_test, y_mean_hat_test, _selectors_test, scaler_dict, inverse_transform=False
    )
    print(f"Done working on group {group}\n", flush=True)

end = time.time()
print(f"Done with everything in {end-start} seconds.", flush=True)