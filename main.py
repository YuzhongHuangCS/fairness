# Data preprocessing
# 1. Add header to data
# 2. Remove unknown: sed '/\?/d' adult.data > adult_known.data

import sklearn
import sklearn.tree
import sklearn.ensemble
import sklearn.metrics
import pandas as pd
import numpy as np
import scipy
import scipy.optimize
import pdb


# because we need to encode categorical feature, have to concate dataframe and then split
df_train_raw = pd.read_csv('adult_known.data', sep=', ', engine='python')
df_test_raw = pd.read_csv('adult_known.test', sep=', ', engine='python')

n_train = len(df_train_raw)
n_test = len(df_test_raw)
df_raw = pd.concat([df_train_raw, df_test_raw])
df = pd.get_dummies(df_raw, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'Y'])

# binrary feature will be mapped to two categories. Remove one of them.
df = df.drop(columns=['sex_Male', 'Y_<=50K'])
X = df.drop(columns=['Y_>50K'])
Y = df['Y_>50K']

X_train = X.iloc[:n_train]
X_test = X.iloc[n_train:]
Y_train = Y.iloc[:n_train]
Y_test = Y.iloc[n_train:]

#clf = sklearn.tree.DecisionTreeClassifier()
clf = sklearn.ensemble.RandomForestClassifier()
clf.fit(X_train, Y_train)
print('Test set accuracy: {}'.format(clf.score(X_test, Y_test)))

Y_pred = clf.predict(X_test)
index_male = (X_test['sex_Female'] == 0)
index_female = (X_test['sex_Female'] == 1)

Y_test_male = Y_test[index_male]
Y_pred_male = Y_pred[index_male]

cf_m= sklearn.metrics.confusion_matrix(Y_test_male, Y_pred_male)
TP_male = cf_m[1, 1] / (cf_m[1, 1] + cf_m[1, 0])
TN_male = cf_m[0, 0] / (cf_m[0, 0] + cf_m[0, 1])
print('TP_male', TP_male)
print('TN_male', TN_male)

Y_test_female = Y_test[index_female]
Y_pred_female = Y_pred[index_female]
cf_f = sklearn.metrics.confusion_matrix(Y_test_female, Y_pred_female)
TP_female = cf_f[1, 1] / (cf_f[1, 1] + cf_f[1, 0])
TN_female = cf_f[0, 0] / (cf_f[0, 0] + cf_f[0, 1])
print('TP_female', TP_female)
print('TN_female', TN_female)


Y_1_female = sum(Y_test_female) / n_test
Y_0_female = 1 - Y_1_female
print('Y_1_female', Y_1_female)
print('Y_0_female', Y_0_female)

Y_1_male = sum(Y_test_male) / n_test
Y_0_male = 1 - Y_1_male
print('Y_1_male', Y_1_male)
print('Y_0_male', Y_0_male)

# initial vale for w
w_initial = 0.5

# I have try alpha=0, 1e-3, 1e-2, 1e-1, 1
# Looks like it have numerical stability issue when alpha is small
alpha = 1

# convention: w for female, 1-w for male
# because we are doing minimization, the second term is flipped
def objective(w):
	obj = alpha * ((w * Y_1_female - (1-w) * Y_1_male)**2 + (w * Y_0_female - (1-w) * Y_0_male)**2) - (w * (TP_female + TN_female) + (1-w) *(TP_male+TN_male))
	return obj

solution = scipy.optimize.minimize(objective, [w_initial], method='TNC', bounds=[(0, 1)])
print(solution)
print('The optimal w is: {}'.format(solution.x))
pdb.set_trace()
print('Pause before exit')
