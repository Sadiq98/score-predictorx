
#importing required libraries
import pandas as pd
import pickle

#loading data
df=pd.read_csv('ipl.csv')

#show data
df.head()

"""# Cleaning Data"""

#Removing unwanted columns
remove_columns=['mid'	,'venue','batsman','bowler','striker','non-striker']
df.drop(labels=remove_columns, axis=1, inplace=True)

#removing unconsistent teams
Consistent_teams=['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
df= df[(df['bat_team'].isin(Consistent_teams)) &(df['bowl_team'].isin(Consistent_teams))]

#Removing first 5 overs data from each match
df=df[df['overs']>=5.0]

df.head(10)

print(df['bat_team'].unique())
print(df['bowl_team'].unique())

#Convert 'date' column from string to datetime object
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

"""# Data Preprocessing"""

#Converting categorial features using oneHot Encoding
#One-hot encoding is essentially the representation of categorical variables as binary vectors.
encoded_df=pd.get_dummies(data=df, columns=['bat_team','bowl_team'])

encoded_df.head()

encoded_df.columns

#Rearranging columns
encoded_df=encoded_df[['date','bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
       'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
       'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
       'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Royal Challengers Bangalore',
       'bowl_team_Sunrisers Hyderabad','runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5',
       'total']]

encoded_df.head(0)

#Splitting data into train and test sets
#Splitting data shows how model performs while performing test on similar data
#'X' is features for input to train test model
#'Y' is expected output label
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]

y_train=encoded_df[encoded_df['date'].dt.year <=2016] ['total'].values
y_test=encoded_df[encoded_df['date'].dt.year >=2017] ['total'].values

X_train.head()

X_test.head()

X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date',axis=True, inplace=True)

X_train.head()

X_test.head()

"""# Building Model"""

#Using regression algo
#Linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Now creating pcikle file
#Pickle converts python object into byte stream to store into file/db
filename = 'score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))