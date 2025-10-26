from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier
from ML_pipeline.Model_evaluation import evaluate_model
from ML_pipeline.Utils import max_val_index

#function to build 4 models and return the best fit model by accuracy measure
def train_model(x_train, x_test, y_train, y_test):
    model_dict = {
        'logistic_reg': LogisticRegression,
        'RandomForest_reg': RandomForestRegressor,
        'DecisionTree_reg': DecisionTreeRegressor,
        'XGBoost': XGBClassifier,
    }

    fitted_model=[]
    roc_auc=[]
    
    for model_name in list(model_dict.keys()):
        if model_name=='XGBoost':
            model= XGBClassifier(use_label_encoder=False)
        else:    
            model = model_dict[model_name]()
        fitted_model.append(model.fit(x_train, y_train))
        roc_auc.append(evaluate_model(y_test, model.predict(x_test), 'roc_auc'))
            
    max_test= max_val_index(roc_auc)
    max_auc=max_test[0]
    max_auc_index = max_test[1]
    final_model=fitted_model[max_auc_index]

    return (final_model, max_auc)
