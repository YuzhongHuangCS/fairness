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
df_train_raw = pd.read_csv('synt_train', sep=', ', engine='python')
df_dev_raw = pd.read_csv('synt_dev', sep=', ', engine='python')
df_test_raw = pd.read_csv('synt_test', sep=', ', engine='python')

n_train = len(df_train_raw)
n_dev = len(df_dev_raw)
n_test = len(df_test_raw)
df_raw = pd.concat([df_train_raw, df_dev_raw ,df_test_raw])

# binrary feature will be mapped to two categories. Remove one of them.
df = pd.get_dummies(df_raw, columns=['gender','needs'])
print(df)
df = df.drop(columns=['gender_male', 'needs_0'])
X = df.drop(columns=['needs_1'])
group_label = X['gender_female']

scaler = sklearn.preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

Y = df['needs_1']

X_train = X_scaled[:n_train]
X_dev = X_scaled[n_train:n_train+n_dev]
X_test = X_scaled[n_train+n_dev:]
Y_train = Y.iloc[:n_train]
Y_dev = Y.iloc[n_train:n_train+n_dev]
Y_test = Y.iloc[n_train+n_dev:]

index_male_train = np.where(group_label[:n_train] == 0)[0].astype(np.int32)
index_female_train = np.where(group_label[:n_train] == 1)[0].astype(np.int32)
index_male_true_train = np.where(np.logical_and(group_label[:n_train] == 0, Y_train==1))[0].astype(np.int32)
index_male_false_train = np.where(np.logical_and(group_label[:n_train] == 0, Y_train==0))[0].astype(np.int32)
index_female_true_train = np.where(np.logical_and(group_label[:n_train] == 1, Y_train==1))[0].astype(np.int32)
index_female_false_train = np.where(np.logical_and(group_label[:n_train] == 1, Y_train==0))[0].astype(np.int32)

index_male_dev = np.where(group_label[n_train:n_train+n_dev] == 0)[0].astype(np.int32)
index_female_dev = np.where(group_label[n_train:n_train+n_dev] == 1)[0].astype(np.int32)
index_male_true_dev = np.where(np.logical_and(group_label[n_train:n_train+n_dev] == 0, Y_dev==1))[0].astype(np.int32)
index_male_false_dev = np.where(np.logical_and(group_label[n_train:n_train+n_dev] == 0, Y_dev==0))[0].astype(np.int32)
index_female_true_dev = np.where(np.logical_and(group_label[n_train:n_train+n_dev] == 1, Y_dev==1))[0].astype(np.int32)
index_female_false_dev = np.where(np.logical_and(group_label[n_train:n_train+n_dev] == 1, Y_dev==0))[0].astype(np.int32)

index_male_test = np.where(group_label[n_train+n_dev:] == 0)[0].astype(np.int32)
index_female_test = np.where(group_label[n_train+n_dev:] == 1)[0].astype(np.int32)
index_male_true_test = np.where(np.logical_and(group_label[n_train+n_dev:] == 0, Y_test==1))[0].astype(np.int32)
index_male_false_test = np.where(np.logical_and(group_label[n_train+n_dev:] == 0, Y_test==0))[0].astype(np.int32)
index_female_true_test = np.where(np.logical_and(group_label[n_train+n_dev:] == 1, Y_test==1))[0].astype(np.int32)
index_female_false_test = np.where(np.logical_and(group_label[n_train+n_dev:] == 1, Y_test==0))[0].astype(np.int32)

# put Y into one hot label
Y_train = np.stack([1-Y_train, Y_train]).T
Y_dev = np.stack([1-Y_dev, Y_dev]).T
Y_test = np.stack([1-Y_test, Y_test]).T

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

# w is the importance of female
# use clip instead of sigmoid to avoid saturation. Make training on w faster
# but it have a problem: once w go out side [0, 1], it will lost graident and couldn't go back
raw_w = tf.Variable(0.5, name='w')
w = tf.clip_by_value(raw_w, 0, 1)

# alpha: importance of imparity loss
# beta: importance of imparity loss + outcome loss
alpha = 0.1
beta = 0.5

L1_output = tf.layers.dense(X_placeholder, DIM_HIDDEN, activation=tf.nn.tanh)
output = tf.layers.dense(L1_output, DIM_OUTPUT, activation=None)

prob = tf.nn.softmax(output)
prob_male = tf.nn.embedding_lookup(prob, index_male_placeholder)
prob_female = tf.nn.embedding_lookup(prob, index_female_placeholder)
prob_male_true = tf.nn.embedding_lookup(prob, index_male_true_placeholder)
prob_male_false = tf.nn.embedding_lookup(prob, index_male_false_placeholder)
prob_female_true = tf.nn.embedding_lookup(prob, index_female_true_placeholder)
prob_female_false = tf.nn.embedding_lookup(prob, index_female_false_placeholder)

loss_imparity = tf.math.squared_difference(w*tf.reduce_mean(prob_female[:, 1]), (1-w)*tf.reduce_mean(prob_male[:, 1])) \
			  + tf.math.squared_difference(w*tf.reduce_mean(prob_female[:, 0]), (1-w)*tf.reduce_mean(prob_male[:, 0]))

label_male = tf.nn.embedding_lookup(Y_placeholder, index_male_placeholder)
label_female = tf.nn.embedding_lookup(Y_placeholder, index_female_placeholder)
output_male = tf.nn.embedding_lookup(output, index_male_placeholder)
output_female = tf.nn.embedding_lookup(output, index_female_placeholder)

'''
loss_outcome = -w*(tf.reduce_mean(prob_female_true[:, 1] + tf.reduce_mean(prob_female_false[:, 0]))) \
			 - (1-w)*(tf.reduce_mean(prob_male_true[:, 1] + tf.reduce_mean(prob_male_false[:, 0]))) \
'''
loss_outcome = w * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_female, logits=output_female)) \
			 + (1-w) * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_male, logits=output_male)) \

pred = tf.math.argmax(prob, axis=1)
diff = tf.to_float(pred) - Y_placeholder[:, 1]
accuracy = 1 - tf.math.reduce_mean(tf.math.abs(diff))
loss_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_placeholder, logits=output))
loss_total = beta * (alpha * loss_imparity + (1-alpha) * loss_outcome) + (1-beta)*loss_entropy

# remove regulization on w
variables = [v for v in tf.trainable_variables() if v != raw_w]
for v in variables:
	loss_total += 1e-4 * tf.nn.l2_loss(v) + 1e-6 * tf.losses.absolute_difference(v, tf.zeros(tf.shape(v)))

lr = tf.Variable(0.01, name='lr', trainable=False)
lr_decay_op = lr.assign(lr * 0.95)
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(loss_total)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	wait = 0
	smallest_loss_total_dev = float('inf')
	patience_lr_decay = 5

	for epoch in range(100000):
		w_train, loss_total_train, loss_entropy_train, accuracy_train, loss_imparity_train, loss_outcome_train, pred_train, train_step = sess.run(
			[w, loss_total, loss_entropy, accuracy, loss_imparity, loss_outcome, pred, train_op],
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

		loss_total_dev, loss_entropy_dev, accuracy_dev, loss_imparity_dev, loss_outcome_dev, pred_dev= sess.run(
			[loss_total, loss_entropy, accuracy, loss_imparity, loss_outcome, pred],
				feed_dict={
					X_placeholder: X_dev,
					Y_placeholder: Y_dev,
					index_male_placeholder: index_male_dev,
					index_female_placeholder: index_female_dev,
					index_male_true_placeholder: index_male_true_dev,
					index_male_false_placeholder: index_male_false_dev,
					index_female_true_placeholder: index_female_true_dev,
					index_female_false_placeholder: index_female_false_dev,
				}
		)

		if loss_total_dev <= smallest_loss_total_dev:
			smallest_loss_total_dev = loss_total_dev
			wait = 0
			print('New smallest')
		else:
			wait += 1
			print('Wait {}'.format(wait))
			if wait % patience_lr_decay == 0:
				sess.run(lr_decay_op)
				print('Apply lr decay, new lr: %f' % lr.eval())
		if(wait==100):
	 		break

		print(f'Epoch: {epoch}, W: {w_train}\ntotal_train: {loss_total_train}, entropy_train: {loss_entropy_train}, accuracy_train: {accuracy_train}, imparity_train: {loss_imparity_train}, outcome_train: {loss_outcome_dev}\ntotal_dev : {loss_total_dev}, entropy_dev : {loss_entropy_dev}, accuracy_dev : {accuracy_dev}, imparity_dev : {loss_imparity_dev}, outcome_dev : {loss_outcome_dev}\n')

	loss_total_test, loss_entropy_test, accuracy_test, loss_imparity_test, loss_outcome_test, pred_test = sess.run(
			[loss_total, loss_entropy, accuracy, loss_imparity, loss_outcome, pred],
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
	print(f'total_test: {loss_total_test}, entropy_test: {loss_entropy_test}, accuracy_test: {accuracy_test}, imparity_test: {loss_imparity_test}, outcome_test: {loss_outcome_test}')
	print("*****************")

	print('===train predictions===')
	print(pred_train)
	print(len(pred_train))
	print(sum(pred_train))

	print('===dev predictions===')
	print(pred_dev)
	print(len(pred_dev))
	print(sum(pred_dev))


	print('===test predictions===')
	print(pred_test)
	print(len(pred_test))
	print(sum(pred_test))

	pdb.set_trace()
	print('Pause before exit')
