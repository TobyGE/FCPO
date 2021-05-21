import pandas as pd
import numpy as np
import ast  

def read_file(filename):
    df = pd.read_csv(filename)
    state = [ast.literal_eval(i) for i in df['state'].values.tolist()]
    user = df['user'].values.tolist()
    history = [np.array(ast.literal_eval(i)) for i in df['history'].values.tolist()]
    
    data = pd.DataFrame ()
    data['user'] = user
    data['state'] = state
    data['history'] = history
    return data

