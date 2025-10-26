from sklearn import preprocessing

#function to change categorial variables to numerical varibales by label encoder or value
def cat_to_num(df, col, method='default', values=None):
    if method == 'default':
        label_encoder = preprocessing.LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])
        return df
    elif method == 'custom':
        for key, val in values.items():
            df[col].loc[df[col] == key] = val
        return df
    else:
        raise ValueError(
            "Only these options for method are allowed : ['default','custom']")
