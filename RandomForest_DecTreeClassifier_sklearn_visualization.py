import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus

inputFile = "PastHires.csv"
df = pd.read_csv(inputFile,header=0)

X = df.iloc[:,:-1].values
y = df.iloc[:,6:7].values

encoder_1 = LabelEncoder()
encoder_2 = LabelEncoder()


X[:,1] = encoder_1.fit_transform(X[:,1])
X[:,3] = encoder_1.fit_transform(X[:,3])
X[:,4] = encoder_1.fit_transform(X[:,4])
X[:,5] = encoder_1.fit_transform(X[:,5])

y = encoder_2.fit_transform(y)

features = list(df.columns[:6])

dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(X,y)

dot_data = StringIO()
tree.export_graphviz(dtc,out_file=dot_data,feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

print('Creating graph......')
graph.write_png('Tree1.png')

rfc = RandomForestClassifier(n_estimators = 5)
rfc = rfc.fit(X,y)

print("Getting Prediction.....")
#Predict employment of an employed 10-year veteran
print (rfc.predict([[10, 1, 4, 0, 0, 0]]))
#...and an unemployed 10-year veteran
print (rfc.predict([[10, 0, 4, 0, 0, 0]]))




