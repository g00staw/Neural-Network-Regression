import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data():
    print("Reading data...")
    dataset = pd.read_csv('data/star_classification.csv')

    dataset = dataset.drop(columns=[
        'obj_ID', 'run_ID', 'rerun_ID', 'cam_col',
        'field_ID', 'spec_obj_ID', 'plate', 'MJD', 'fiber_ID'
    ])

    dataset['u_g'] = dataset['u'] - dataset['g']
    dataset['g_r'] = dataset['g'] - dataset['r']
    dataset['r_i'] = dataset['r'] - dataset['i']
    dataset['i_z'] = dataset['i'] - dataset['z']

    le = LabelEncoder()
    dataset['class'] = le.fit_transform(dataset['class'])

    X = dataset.drop('class', axis=1)
    y = dataset['class']

    return X, y, le
