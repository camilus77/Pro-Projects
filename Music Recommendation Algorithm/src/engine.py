#importing required packages
from sklearn.model_selection import train_test_split
from ML_pipeline.Utils import read_data
from ML_pipeline.Utils import rename_column
from ML_pipeline.Utils import separate_date_col
from ML_pipeline.Utils import con_date_in_str
from ML_pipeline.Utils import counting_value
from ML_pipeline.Utils import merge_dataframes
from ML_pipeline.Impute import impute
from ML_pipeline.Utils import max_val_index
from ML_pipeline.Utils import drop_col
from ML_pipeline.Cat_to_num import cat_to_num
from ML_pipeline.Train_model import train_model
from ML_pipeline.Feature_importance import feature_importance
from ML_pipeline.Outlier import detect_outliers
from ML_pipeline.Outlier import remove_outliers
import joblib

#reading the data
train_music_1=read_data("../input/train.csv")
songs_data_1=read_data("../input/songs.csv")
member_data_1=read_data("../input/members.csv")

#taking chunk of data to run the code
#one can take entire data and run
train_music=train_music_1[0:10000]
songs_data=songs_data_1[0:10000]
member_data=member_data_1[0:10000]

#renaming the coulmns in train_music and members_data
train_music=rename_column(train_music, 'msno', 'user_id')
member_data=rename_column(member_data, 'msno', 'user_id')

#converting regsitration and expiration date into string
member_data=con_date_in_str(member_data,'registration_init_time','registration_str')
member_data=con_date_in_str(member_data,'expiration_date','expiration_str')

#extracting year, month and column from the string in registration_init_timr and expiration_date in member-data
member_data=separate_date_col(member_data, 'registration_str', ['registration_year','registration_month','registration_date'])
member_data=separate_date_col(member_data, 'expiration_str', ['expiration_year','expiration_month','expiration_day'])

#dropping registration_init_timr and expiration_date columns from member-data 
member_data=drop_col(member_data, ['registration_init_time','expiration_date','registration_str','expiration_str'])

#checking and removing outliers in members data
outliers_age=detect_outliers(member_data, 'bd', 3)
thresh_age= min(outliers_age)-1
member_data=remove_outliers(member_data, 'bd', thresh_age)

#Counting number of values in particular entry and assigning them to a new column in songs data
songs_data=songs_data.copy()
songs_data[['count_of_genre_ids']] = counting_value(songs_data['genre_ids'])
songs_data[['count_of_artist']] = counting_value(songs_data['artist_name'])
songs_data[['count_of_composer']] = counting_value(songs_data['composer'])
songs_data[['count_of_lyricist']] = counting_value(songs_data['lyricist'])


#mereging all the datasets for model building
merged_music_data = merge_dataframes(train_music, member_data, 'user_id')
merged_music_data= merge_dataframes(merged_music_data,songs_data, 'song_id')

#The methods available for imputation are mean, mode, median, value, missing, drop
#Imputing missing values by mode
#replacing string varibales with numeric values
mode_merged_data = merged_music_data.copy()
for col in mode_merged_data.columns:
    mode_merged_data=impute(mode_merged_data, col, 'mode')
    mode_merged_data=cat_to_num(mode_merged_data, col,'default')

#Dropping the missing values
#replacing string varibales with numeric values
removed_null_data = merged_music_data.copy()
for col in removed_null_data.columns:
    removed_null_data=impute(removed_null_data, col, 'drop')
    removed_null_data=cat_to_num(removed_null_data, col,'default')

#Making a new label as missing
#replacing string varibales with numeric values
missing_label_merged_data = merged_music_data.copy()
for col in missing_label_merged_data.columns:
    missing_label_merged_data=impute(missing_label_merged_data, col, 'missing')
    missing_label_merged_data=cat_to_num(missing_label_merged_data, col,'default')

# Train test split 80-20
x_train_mode, x_test_mode, y_train_mode, y_test_mode = train_test_split(mode_merged_data.drop(['target']
                                                ,axis=1),mode_merged_data['target'],test_size=0.20,random_state=40)

x_train_removed_null, x_test_removed_null,y_train_removed_null, y_test_removed_null = train_test_split(removed_null_data.drop(['target'],
                                                        axis=1),removed_null_data['target'],test_size=0.20,random_state=40)

x_train_missing_label, x_test_missing_label, y_train_missing_label, y_test_missing_label = train_test_split(missing_label_merged_data.drop(['target'],axis=1),
                 missing_label_merged_data['target'],test_size=0.20,random_state=40)


#Model Building and evaluation
best_models=[]
accuracy=[]
test1=train_model(x_train_mode, x_test_mode, y_train_mode, y_test_mode)
best_models.append(test1[0])
accuracy.append(test1[1])

test2=train_model(x_train_removed_null, x_test_removed_null, y_train_removed_null, y_test_removed_null)
best_models.append(test2[0])
accuracy.append(test2[1])

test3=train_model(x_train_missing_label, x_test_missing_label, y_train_missing_label, y_test_missing_label)
best_models.append(test3[0])
accuracy.append(test3[1])

#choosing best model out of above 3 models
final_accuracy,index=max_val_index(accuracy)
final_model=best_models[index]

#dumping the model in output folder
joblib.dump(final_model, '../output/best_fitted_model.pkl' )
print('Best Model saved as pkl file in '+str('../output/best_fittsed-model.pkl'))

#Feature importance
model = joblib.load('../output/best_fitted_model.pkl')
fi_imp_missing= feature_importance(x_train_mode.columns, model)