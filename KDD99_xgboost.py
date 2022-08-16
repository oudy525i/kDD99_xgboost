import numpy as np
import pandas as pd
import pickle

from scoring import cost_based_scoring
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
featuresList = ("duration",  
                "protocol_type",  
                "service", 
                "flag", 
                "src_bytes",  
                "dst_bytes",  
                "land", 
                "wrong_fragment",  
                "urgent", 
                "hot",  
                "num_failed_logins",  
                "logged_in",  
                "num_compromised",  
                "root_shell",  
                "su_attempted",  
                "num_root",  
                "num_file_creations", 
                "num_shells",  
                "num_access_files",  
                "num_outbound_cmds",  
                "is_host_login",  
                "is_guest_login",  
        
                "count",  
                "srv_count",  
                "serror_rate",  
                "srv_serror_rate",  
                "rerror_rate",  
                "same_srv_rate",  
                "diff_srv_rate", 
                "srv_diff_host_rate",  
                "dst_host_count",
                "dst_host_srv_count",
                "dst_host_same_srv_rate",
                "dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate",
                "dst_host_serror_rate",
                "dst_host_srv_serror_rate",
                "dst_host_rerror_rate",
                "dst_host_srv_rerror_rate",
                # ----------
                # category
                "attack_type"
                )

df = pd.read_csv(r'data/train10pc', header=None, names=__ATTR_NAMES)

df = processing.merge_sparse_feature(df)

df = processing.one_hot(df)

df = processing.map2major5(df)
with open(r'data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("training data loaded")

y = df["attack_type"]
X = df[selected_feat_names].values


xgboostc = XGBClassifier()
xgboostc = xgboostc.fit(X, y)
print("training finished")

df=pd.read_csv(r'data/corrected',header=None, names=__ATTR_NAMES)
df = processing.merge_sparse_feature(df)

df = processing.one_hot(df)

df = processing.map2major5(df)
with open(r'data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("test data loaded")

X = df[selected_feat_names].values
y = df['attack_type'].values
y_rf = xgboostc.predict(X)

print("xgbdt results:")
cost_based_scoring.score(y, y_rf, True)



