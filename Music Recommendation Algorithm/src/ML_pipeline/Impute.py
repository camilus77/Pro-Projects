#function to impute missing values by differnt methods
def impute(df, col, method, value=0):
    if method == 'mean':
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
        return df
    elif method == 'median':
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        return df
    elif method == 'mode':
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        return df
    elif method == 'missing':
        df[col] = df[col].fillna('missing')
        return df
    elif method == 'drop':
        df = df.dropna()
        return df
    elif method == 'value':
        df[col] = df[col].fillna(value)
        return df
    else:
        raise ValueError(
            "Only these options for method are allowed : ['mean','median','mode', 'missing', 'drop','value']")

