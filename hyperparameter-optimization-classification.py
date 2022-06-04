import optuna
from optuna import Trial as trial
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
optuna.logging.set_verbosity(optuna.logging.INFO)
from sklearn.model_selection import train_test_split
# import lightgbm as lgb 
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
# https://github.com/optuna/optuna-examples


def objective(trial):
    
    # list of classifiers
    classifier_name = trial.suggest_categorical("classifier", ["Catboost"]) #["LR","SVC", "RandomForest", "XGBoost","Catboost"])
    
    # logistic regression
    if classifier_name == "LR":
        params = {
                    "C"      : trial.suggest_float("C", 1e-10, 1e10, log=True),
                    "class_weight" : trial.suggest_categorical('class_weight', ['balanced']),
                    "max_iter" :  trial.suggest_int('max_iter', 100, 500),
                    "random_state" : seed
                    }
        
        classifier_obj = LogisticRegression(**params)     
    
    # svc
    if classifier_name == "SVC":
        params = {
                    "C"      : trial.suggest_float("C", 1e-10, 1e10, log=True),
                    "kernel" : trial.suggest_categorical('kernel', ['linear','poly','rbf','sigmoid']),
                    "gamma"  : trial.suggest_categorical("gamma", ['scale', 'auto']),
                    "random_state" : seed
                    }

        classifier_obj = SVC(**params)        

    # rf
    elif classifier_name == "RandomForest":
        params = {
                    "class_weight":   trial.suggest_categorical('class_weight',["balanced"]),
                    "bootstrap" :  trial.suggest_categorical('bootstrap',[True, False]),
                    "criterion" : trial.suggest_categorical('criterion', ['gini', 'entropy']),
                    "max_depth" : trial.suggest_int('max_depth', 2, 16),
                    "max_features" : trial.suggest_categorical('max_features', ['auto','log2',None]),
                    "max_leaf_nodes" : trial.suggest_int('max_leaf_nodes', 1, 100),
                    "n_estimators" :  trial.suggest_int('n_estimators', 300, 2000),
                    "random_state" : seed,
                    "n_jobs" : 2
                    }
        
        if params["bootstrap"] == True:
            params["oob_score"] = True
        else:
            params["oob_score"] = False
        
        classifier_obj = RandomForestClassifier(**params)
  
    # xgb
    elif classifier_name == "XGBoost":
        params = {
                    'tree_method':'gpu_hist',
                    'predictor': 'gpu_predictor',
                    "verbosity": 0,
                    "objective": "multi:softmax", #"binary:logistic",
                    # use exact for small dataset.
                    "tree_method": "exact",
                    # defines booster, gblinear for linear functions.
                    "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                    # L2 regularization weight.
                    "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                    # L1 regularization weight.
                    "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                    # sampling ratio for training data.
                    "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                    # sampling according to each tree.
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                    }

        if params["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            params["max_depth"] = trial.suggest_int("max_depth", 3, 15, step=2)
            # minimum child weight, larger the term more conservative the tree.
            params["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 30)
            params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if params["booster"] == "dart":
            params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
            
       
        classifier_obj = XGBClassifier(**params)
        
    elif classifier_name == "Catboost":
        params = {
        # "learning_rate": defined by loss function automatically trial.suggest_float("learning_rate", 1e-4, 1.0, log=True),
        # "colsample_bylevel": only CPU supported trial.suggest_float("colsample_bylevel", 0.01, 0.1), 
        # "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]), only PLAIN is GPU supported
        #'eval_metric': "MultiClass", #"Accuracy",#'',
        # 'loss_function': 'MultiClass',

        "loss_function": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        'eval_metric' : "F1",
        "n_estimators": 2, #trial.suggest_int('n_estimators', 1000, 5000, 500), # 1000 default, #
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli",  "Poisson"]), #"MVS" only CPU support
        "random_strength" : trial.suggest_int("random_strength", 0,1),
        'od_wait':trial.suggest_int('od_wait', 500, 2300),
        "border_count": trial.suggest_int("border_count", 128, 254),
        "min_data_in_leaf" : trial.suggest_int("min_data_in_leaf", 1, 10), 
        "random_seed":seed,
        "verbose": True,
        'task_type':"GPU",
        'devices': '0:1',
                    }
        
        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.1, 1)
            
        classifier_obj = CatBoostClassifier(**params)
        
        
    score = cross_val_score(classifier_obj, train_x, train_y, cv=3, scoring="balanced_accuracy",n_jobs=2)
    mean_acc = score.mean()

    return mean_acc        
  
  
#Execute optuna and set hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10, gc_after_trial=True, timeout=600, show_progress_bar=True)

print(study.best_trial, '\n')
print('Accuracy: {}'.format(study.best_trial.value),'\n')
print("Best hyperparameters: {}".format(study.best_trial.params))


optuna.visualization.plot_optimization_history(study)

optuna.visualization.plot_slice(study)

optuna.visualization.plot_param_importances(study)

optuna.visualization.plot_parallel_coordinate(study)
