# Data preprocessing
# 1. Add header to data
# 2. Remove unknown: sed '/\?/d' adult.data > adult_known.data
# 3. Remove tailing . in "adult.test": sed 's/.$//' adult_known.test > adult_known.test2

import sklearn
import sklearn.tree
import sklearn.ensemble
import sklearn.metrics
import pandas as pd
import numpy as np
import scipy
import scipy.optimize
import pdb
import tensorflow as tf

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

index_male_train = np.where(X_train['sex_Female'] == 0)[0].astype(np.int32)
index_female_train = np.where(X_train['sex_Female'] == 1)[0].astype(np.int32)
index_male_true_train = np.where(np.logical_and(X_train['sex_Female'] == 0, Y_train==1))[0].astype(np.int32)
index_male_false_train = np.where(np.logical_and(X_train['sex_Female'] == 0, Y_train==0))[0].astype(np.int32)
index_female_true_train = np.where(np.logical_and(X_train['sex_Female'] == 1, Y_train==1))[0].astype(np.int32)
index_female_false_train = np.where(np.logical_and(X_train['sex_Female'] == 1, Y_train==0))[0].astype(np.int32)

index_male_test = np.where(X_test['sex_Female'] == 0)[0].astype(np.int32)
index_female_test = np.where(X_test['sex_Female'] == 1)[0].astype(np.int32)
index_male_true_test = np.where(np.logical_and(X_test['sex_Female'] == 0, Y_test==1))[0].astype(np.int32)
index_male_false_test = np.where(np.logical_and(X_test['sex_Female'] == 0, Y_test==0))[0].astype(np.int32)
index_female_true_test = np.where(np.logical_and(X_test['sex_Female'] == 1, Y_test==1))[0].astype(np.int32)
index_female_false_test = np.where(np.logical_and(X_test['sex_Female'] == 1, Y_test==0))[0].astype(np.int32)

Y_train = np.stack([Y_train, 1-Y_train]).T
Y_test = np.stack([Y_test, 1-Y_test]).T

DIM_INPUT = X_train.shape[1]
DIM_HIDDEN = 256
DIM_OUTPUT = 2

X_placeholder = tf.placeholder(tf.float32, [None, DIM_INPUT])
Y_placeholder = tf.placeholder(tf.float32, [None, DIM_OUTPUT])
index_male_placeholder = tf.placeholder(tf.int32, [None])
index_female_placeholder = tf.placeholder(tf.int32, [None])
index_male_true_placeholder = tf.placeholder(tf.int32, [None])
index_male_false_placeholder = tf.placeholder(tf.int32, [None])
index_female_true_placeholder = tf.placeholder(tf.int32, [None])
index_female_false_placeholder = tf.placeholder(tf.int32, [None])

W1 = tf.get_variable('weight1', shape=(DIM_INPUT, DIM_HIDDEN), initializer=tf.glorot_uniform_initializer())
b1 = tf.get_variable('bias1', shape=(1, DIM_HIDDEN), initializer=tf.zeros_initializer())
W2 = tf.get_variable('weight2', shape=(DIM_HIDDEN, DIM_OUTPUT), initializer=tf.glorot_uniform_initializer())
b2 = tf.get_variable('bias2', shape=(1, DIM_OUTPUT), initializer=tf.zeros_initializer())
w_raw = tf.Variable(0.0)
w = tf.math.sigmoid(w_raw)

# alpha: importance of imparity loss
# beta: importance of imparity loss + outcome loss
alpha = 0.01
beta = 0.99

output = tf.matmul(tf.nn.relu(tf.matmul(X_placeholder, W1) + b1), W2) + b2
prob = tf.nn.softmax(output)

prob_male = tf.nn.embedding_lookup(prob, index_male_placeholder)
prob_female = tf.nn.embedding_lookup(prob, index_female_placeholder)
prob_male_true = tf.nn.embedding_lookup(prob, index_male_true_placeholder)
prob_male_false = tf.nn.embedding_lookup(prob, index_male_false_placeholder)
prob_female_true = tf.nn.embedding_lookup(prob, index_female_true_placeholder)
prob_female_false = tf.nn.embedding_lookup(prob, index_female_false_placeholder)

loss_imparity = tf.math.squared_difference(w*tf.reduce_mean(prob_female[:, 1]), (1-w)*tf.reduce_mean(prob_male[:, 1])) \
			  + tf.math.squared_difference(w*tf.reduce_mean(prob_female[:, 0]), (1-w)*tf.reduce_mean(prob_male[:, 0]))

loss_outcome = -w*(tf.reduce_mean(prob_female_true[:, 1] + tf.reduce_mean(prob_female_false[:, 0]))) \
			 - (1-w)*(tf.reduce_mean(prob_male_true[:, 1] + tf.reduce_mean(prob_male_false[:, 0]))) \

pred = tf.math.argmax(prob, axis=1)
diff = tf.to_float(pred) - Y_placeholder[:, 1]
accuracy = 1 - tf.math.reduce_mean(tf.math.abs(diff))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_placeholder, logits=output))
loss_reg = (1-beta)*loss + beta * (alpha * loss_imparity + (1-alpha) * loss_outcome)

for v in tf.trainable_variables():
	loss_reg += 5e-4 * tf.nn.l2_loss(v) + 1e-8 * tf.losses.absolute_difference(v, tf.zeros(tf.shape(v)))

optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss_reg)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for epoch in range(1000):
		loss_train, w_train, loss_imparity_train, accuracy_train, pred_train, _train_step = sess.run(
			[loss, w, loss_imparity, accuracy, pred, train_op],
				feed_dict={
					X_placeholder: X_train,
					Y_placeholder: Y_train,
					index_male_placeholder: index_male_train,
					index_female_placeholder: index_female_train,
					index_male_true_placeholder: index_male_true_train,
					index_male_false_placeholder: index_male_false_train,
					index_female_true_placeholder: index_female_true_train,
					index_female_false_placeholder: index_female_false_train,
				}
		)

		loss_test, loss_imparity_test, accuracy_test = sess.run(
			[loss, loss_imparity, accuracy],
				feed_dict={
					X_placeholder: X_test,
					Y_placeholder: Y_test,
					index_male_placeholder: index_male_test,
					index_female_placeholder: index_female_test,
					index_male_true_placeholder: index_male_true_test,
					index_male_false_placeholder: index_male_false_test,
					index_female_true_placeholder: index_female_true_test,
					index_female_false_placeholder: index_female_false_test,
				}
		)

		print(epoch, w_train, loss_train, loss_imparity_train, accuracy_train, sklearn.metrics.accuracy_score(Y_train[:, 1], pred_train), loss_test, loss_imparity_test, accuracy_test)
	pdb.set_trace()
	print('OK')
