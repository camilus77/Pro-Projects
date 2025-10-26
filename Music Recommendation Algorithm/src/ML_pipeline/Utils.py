import pandas as pd

# function to read the data
def read_data(file_path, **kwargs):
    raw_data=pd.read_csv(file_path  ,**kwargs)
    return raw_data


#function to rename particuar columns in data
def rename_column(df, col, rename_col): 
    if rename_col in df.columns:
        raise ValueError(
            "renamed column already exist")
    else:
            df= df.rename(columns={col:rename_col}) 
    return df  

#function to convert date integer column into string column
def con_date_in_str(df,date_col,new_str_col):
    if new_str_col in df:
        raise KeyError(
                f"{new_str_col} column already exists. Please enter a different value for new_col_name")
    df[[new_str_col]] = df[[date_col]].applymap(str).applymap(lambda s: "{}-{}-{}".format(s[4:6],s[6:], s[0:4]))
    return df

#function to separate date columns in year month and day
def separate_date_col(df, date_col, new_col_name):
    for col in new_col_name:
        if col in df:
            raise KeyError(
                f"{col} column already exists. Please enter a different value for new_col_name")
    df[new_col_name[0]] = pd.DatetimeIndex(df[date_col]).year
    df[new_col_name[1]] = pd.DatetimeIndex(df[date_col]).month
    df[new_col_name[2]] = pd.DatetimeIndex(df[date_col]).day
    return df

#function to drop columns from data
def drop_col(df, col_list):
    for col in col_list:
        if col not in df.columns:
            raise ValueError(
                f"Column does not exit in dataframe")
    df=df.drop(col_list, axis=1)
    return(df)

#function to count values in particular entry
def counting_value(data):
    count = []
    for ids in data:
        try:
            if len(ids) != 0:
                count_ids = 1
                for i in ids:
                    if i == '|':
                        count_ids+=1
                count.append(count_ids)
        except TypeError:
            count.append(0)
    return count

#function to merge 2 dataframes
def merge_dataframes(df1, df2, col_name):
    combined_data = pd.merge(df1, df2, on=col_name)
    return combined_data

#function to find maximum value and returning maximum value and its index
def max_val_index(l):
    max_l=max(l)
    max_index=l.index(max_l)
    return (max_l, max_index)

      