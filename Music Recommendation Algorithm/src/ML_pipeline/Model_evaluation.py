from sklearn.metrics import roc_auc_score

#function to evaluate model by using accuracy measures
def evaluate_model(y_test, y_pred, method):
    if method=='roc_auc':
        score= roc_auc_score(y_test, y_pred)
    else: 
        print("Only available acuuracy measures is roc_auc.")
    return score
