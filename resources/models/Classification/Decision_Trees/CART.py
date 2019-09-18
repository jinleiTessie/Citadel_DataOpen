import itertools
import random 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

#input data
training_data = pd.DataFrame([['low',  'sunny',   'yes','yes'],
    ['high', 'sunny',   'yes','no'],
    ['med',  'cloudy',  'yes','no'],
    ['low',  'raining', 'yes','no'],
    ['low',  'cloudy',  'no' ,'yes'],
    ['high', 'sunny',   'no' ,'no'],
    ['high', 'raining', 'no' ,'no'],
    ['med',  'cloudy',  'yes','no'],
    ['low',  'raining', 'yes','no'],
    ['low',  'raining', 'no' ,'yes'],
    ['med',  'sunny',   'no' ,'yes'],
    ['high', 'sunny',   'yes','no']], columns=["supplies","weather","worked?","went shopping?"])

#change categorical data to numeric representation, need encoder
classes = {#define all possible values in each category
    'supplies': ['low', 'med', 'high'],
    'weather':  ['raining', 'cloudy', 'sunny'],
    'worked?':  ['yes', 'no']}
#testing data: random choose values from each category
prediction_data = []
for _ in itertools.repeat(None, 5):
    prediction_data.append([ # random choose values from each category
        random.choice(classes['supplies']),
        random.choice(classes['weather']),
        random.choice(classes['worked?'])])
prediction_data=pd.DataFrame(prediction_data, columns=["supplies","weather","worked?"])    
#code data #output data: do not need encode
encoder = OneHotEncoder(categories=[classes['supplies'], classes['weather'], classes['worked?']])
training_input_encoded =encoder.fit_transform(training_data[["supplies","weather","worked?"]])

# fit decision tree model
classifier = DecisionTreeClassifier()
tree = classifier.fit(training_input_encoded, training_data[["went shopping?"]])

#prediction
prediction_data_encoded=encoder.fit_transform(prediction_data)
prediction_results = tree.predict(prediction_data_encoded)
print (prediction_results)
