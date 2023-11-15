import pandas as pd
import numpy as np


# This function takes in a df with column headers in the following format:
# smiles, feature1, feature2, prop, value


#recently changed smiles to poly_smiles (change back after no longer necessary)
def convert2dict(df, removeCols, physics_cols, isMLP):
    removeCols.append('prop')
    removeCols.append('value')
    removeCols.append('smiles')
    allCols = list(df)


    allCols = [e for e in allCols if e not in removeCols]
    allCols = [e for e in allCols if e not in physics_cols]

    num_graph_feats = len(allCols)


    newCols = ['smiles', 'physics_feats', 'graph_feats', 'prop', 'value']

    if isMLP:
        newCols.remove('smiles')

    new_df = pd.DataFrame(columns = newCols)

    graph_feats = []
    for row in range(len(df)):
        vals = {}
        dfRow = df.iloc[row]
        for col in allCols:
            vals[col] = dfRow[col]
        graph_feats.append(vals)

    physics_feats = []
    for row in range(len(df)):
        vals = {}
        dfRow = df.iloc[row]
        for col in physics_cols:
            vals[col] = dfRow[col]
        physics_feats.append(vals)
    
    for col in newCols:
        if col == 'graph_feats':
            new_df[col] = graph_feats
        elif col == 'physics_feats':
            new_df[col] = physics_feats
        else:
            new_df[col] = list(df[col])

    #print(new_df.sample(n=10))

    return new_df, num_graph_feats


# Testing the code  

#df = pd.read_csv('fp_computational_MLPgnn.csv')
#removeCols = ['__fingerprint_success__', 'index']
#master_data = convert2dict(df,removeCols, True)
#print(master_data.sample(n=10))
#print("\n")
