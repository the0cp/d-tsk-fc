import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_kddcup99(file_path, test_size=0.25):
    """
    1. load KDD CUP 1999 with pandas.
    2. One-hot encode categorical features.
    3. Min-Max normalize numerical features.
    4. encodes labels to integers.
    5. split dataset
    6. One-hot encode training labels.

    Args:
    - file_path (str): KDD Cup 1999 path
    - test_size (float): split ratio for test set

    Returns:
    - X_train, T_train, X_test, y_test, num_classes
    """
    print(f"Loading data from: {file_path}")
    
    # Add column names
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label'
    ]
    
    df = pd.read_csv(file_path, header=None, names=column_names).sample(frac=0.005, random_state=42)
    

    label_counts = df['label'].value_counts()
    rare_classes = label_counts[label_counts < 2].index
    if not rare_classes.empty:
        print(f"Found {len(rare_classes)} rare classes with only 1 sample: {list(rare_classes)}")
        df = df[~df['label'].isin(rare_classes)]
        print(f"Removed rare class samples. New dataset size: {len(df)} records.")


    # extract labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    print("Preprocessing data...")
    X_processed = pd.get_dummies(X, columns=categorical_cols, dtype=float)
    
    # label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    numerical_cols = X.select_dtypes(include=np.number).columns
    scaler = MinMaxScaler()
    
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    T_train = np.eye(num_classes)[y_train]
    
    print("Preprocessing complete.")
    
    return X_train.to_numpy(), T_train, X_test.to_numpy(), y_test, num_classes