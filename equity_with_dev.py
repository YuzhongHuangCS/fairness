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

tf.random.set_random_seed(2019)

# because we need to encode categorical feature, have to concate dataframe and then split
#For Adult Dataset
# df_train_raw = pd.read_csv('adult_known.data', sep=', ', engine='python')
# df_dev_raw = pd.read_csv('adult_nina_dev', sep=', ', engine='python')
# df_test_raw = pd.read_csv('adult_nina_test', sep=', ', engine='python')

#For Compas Dataset
# colnames = ['sex','age_cat','race','juv_fel_count','juv_misd_count','juv_other_count','priors_count','c_charge_degree','is_recid','score_text','two_year_recid']
# df_train_raw = pd.read_csv('compas_train.csv',names=colnames,encoding = "ISO-8859-1")
# df_dev_raw = pd.read_csv('compas_dev.csv',names=colnames,encoding = "ISO-8859-1")
# df_test_raw = pd.read_csv('compas_test.csv', names=colnames,encoding = "ISO-8859-1")

#For German Dataset
df_train_raw = pd.read_csv('pre_processed_german_train', sep=' ', engine='python')
df_dev_raw = pd.read_csv('pre_processed_german_dev', sep=' ', engine='python')
df_test_raw = pd.read_csv('pre_processed_german_test', sep=' ', engine='python')

n_train = len(df_train_raw)
n_dev = len(df_dev_raw)
n_test = len(df_test_raw)
df_raw = pd.concat([df_train_raw, df_dev_raw,df_test_raw])

#For Adult Dataset
#df = pd.get_dummies(df_raw, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'Y'])

#For Compas Dataset
#df = pd.get_dummies(df_raw, columns=['sex','age_cat','race','c_charge_degree','score_text'])

#For german dataset
df = pd.get_dummies(df_raw, columns=['checking_acc','credit_history','purpose','saving_acc','employment','sex','debtors','property','plans','housing','job','telephone','foreign'])

# binrary feature will be mapped to two categories. Remove one of them.
#For Adult Dataset
# df = df.drop(columns=['sex_Male', 'Y_<=50K'])
# X = df.drop(columns=['Y_>50K'])
# group_label = X['sex_Female']

#For Compas Dataset
# df = df.drop(columns=['sex_Male'])
# X = df.drop(columns=['two_year_recid'])
# group_label = X['sex_Female']

#For German dataset
df = df.drop(columns=['sex_Male'])
X = df.drop(columns=['label'])
group_label = X['sex_Female']

scaler = sklearn.preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
#For Adult Dataset
#Y = df['Y_>50K']

#For Compas Dataset
#Y = df['two_year_recid']

#For german dataset
Y = df['label']

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

train_data_one_female_prob = group_label[:n_train][index_female_true_train].shape[0]/index_female_train.shape[0]
train_data_zero_female_prob = group_label[:n_train][index_female_false_train].shape[0] /index_female_train.shape[0]
train_data_one_male_prob = group_label[:n_train][index_male_true_train].shape[0]/index_male_train.shape[0]
train_data_zero_male_prob = group_label[:n_train][index_male_false_train].shape[0]/index_male_train.shape[0]

dev_data_one_female_prob = group_label[n_train:n_train+n_dev][index_female_true_dev].shape[0]/index_female_dev.shape[0]
dev_data_zero_female_prob = group_label[n_train:n_train+n_dev][index_female_false_dev].shape[0]/index_female_dev.shape[0]
dev_data_one_male_prob = group_label[n_train:n_train+n_dev][index_male_true_dev].shape[0]/index_male_dev.shape[0]
dev_data_zero_male_prob = group_label[n_train:n_train+n_dev][index_male_false_dev].shape[0]/index_male_dev.shape[0]

test_data_one_female_prob = group_label[n_train+n_dev:][index_female_true_test].shape[0]/index_female_test.shape[0]
test_data_zero_female_prob = group_label[n_train+n_dev:][index_female_false_test].shape[0]/index_female_test.shape[0]
test_data_one_male_prob = group_label[n_train+n_dev:][index_male_true_test].shape[0]/index_male_test.shape[0]
test_data_zero_male_prob = group_label[n_train+n_dev:][index_male_false_test].shape[0]/index_male_test.shape[0]


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
data_female_one_placeholder = tf.placeholder(tf.float32)
data_female_zero_placeholder = tf.placeholder(tf.float32)
data_male_one_placeholder = tf.placeholder(tf.float32)
data_male_zero_placeholder = tf.placeholder(tf.float32)

# w is the importance of female
# use clip instead of sigmoid to avoid saturation. Make training on w faster
# but it have a problem: once w go out side [0, 1], it will lost graident and couldn't go back
raw_w = tf.Variable(0.5, name='w')
w = tf.clip_by_value(raw_w, 0, 1)

# alpha: importance of imparity loss
# beta: importance of imparity loss + outcome loss
alpha = 0.1
beta = 0.9

L1_output = tf.layers.dense(X_placeholder, DIM_HIDDEN, activation=tf.nn.tanh)
output = tf.layers.dense(L1_output, DIM_OUTPUT, activation=None)

prob = tf.nn.softmax(output)
prob_male = tf.nn.embedding_lookup(prob, index_male_placeholder)
prob_female = tf.nn.embedding_lookup(prob, index_female_placeholder)
prob_male_true = tf.nn.embedding_lookup(prob, index_male_true_placeholder)
prob_male_false = tf.nn.embedding_lookup(prob, index_male_false_placeholder)
prob_female_true = tf.nn.embedding_lookup(prob, index_female_true_placeholder)
prob_female_false = tf.nn.embedding_lookup(prob, index_female_false_placeholder)

fairness_loss = tf.math.squared_difference(tf.reduce_mean(prob_female[:, 1])+data_female_one_placeholder, tf.reduce_mean(prob_male[:, 1])+data_male_one_placeholder) \
			  + tf.math.squared_difference(tf.reduce_mean(prob_female[:, 0])+data_female_zero_placeholder, tf.reduce_mean(prob_male[:, 0])+data_male_zero_placeholder)
#fairness_loss = tf.math.squared_difference(tf.reduce_mean(prob_female[:, 1]), tf.reduce_mean(prob_male[:, 1])) \
#			  + tf.math.squared_difference(tf.reduce_mean(prob_female[:, 0]), tf.reduce_mean(prob_male[:, 0]))

label_male = tf.nn.embedding_lookup(Y_placeholder, index_male_placeholder)
label_female = tf.nn.embedding_lookup(Y_placeholder, index_female_placeholder)
output_male = tf.nn.embedding_lookup(output, index_male_placeholder)
output_female = tf.nn.embedding_lookup(output, index_female_placeholder)

pred = tf.math.argmax(prob, axis=1)
diff = tf.to_float(pred) - Y_placeholder[:, 1]
accuracy = 1 - tf.math.reduce_mean(tf.math.abs(diff))
loss_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_placeholder, logits=output))
#loss_total = beta * (fairness_loss) + (1-beta)*loss_entropy
loss_total = loss_entropy

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

	for epoch in range(10000):
		w_train, loss_total_train, loss_entropy_train, accuracy_train, fairness_loss_train, pred_train, train_step = sess.run(
			[w, loss_total, loss_entropy, accuracy, fairness_loss,  pred, train_op],
				feed_dict={
					X_placeholder: X_train,
					Y_placeholder: Y_train,
					index_male_placeholder: index_male_train,
					index_female_placeholder: index_female_train,
					index_male_true_placeholder: index_male_true_train,
					index_male_false_placeholder: index_male_false_train,
					index_female_true_placeholder: index_female_true_train,
					index_female_false_placeholder: index_female_false_train,
					data_female_one_placeholder: train_data_one_female_prob,
					data_female_zero_placeholder: train_data_zero_female_prob,
					data_male_one_placeholder: train_data_one_male_prob,
					data_male_zero_placeholder: train_data_zero_male_prob
				}
		)

		loss_total_dev, loss_entropy_dev, accuracy_dev, fairness_loss_dev, pred_dev= sess.run(
			[loss_total, loss_entropy, accuracy, fairness_loss, pred],
				feed_dict={
					X_placeholder: X_dev,
					Y_placeholder: Y_dev,
					index_male_placeholder: index_male_dev,
					index_female_placeholder: index_female_dev,
					index_male_true_placeholder: index_male_true_dev,
					index_male_false_placeholder: index_male_false_dev,
					index_female_true_placeholder: index_female_true_dev,
					index_female_false_placeholder: index_female_false_dev,
					data_female_one_placeholder: dev_data_one_female_prob,
					data_female_zero_placeholder: dev_data_zero_female_prob,
					data_male_one_placeholder: dev_data_one_male_prob,
					data_male_zero_placeholder: dev_data_zero_male_prob
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

		print(f'Epoch: {epoch}, W: {w_train}\ntotal_train: {loss_total_train}, entropy_train: {loss_entropy_train}, accuracy_train: {accuracy_train}, imparity_train: {fairness_loss_train}\ntotal_dev : {loss_total_dev}, entropy_dev : {loss_entropy_dev}, accuracy_dev : {accuracy_dev}, imparity_dev : {fairness_loss_dev}\n')

	loss_total_test, loss_entropy_test, accuracy_test, fairness_loss_test, pred_test = sess.run(
			[loss_total, loss_entropy, accuracy, fairness_loss,  pred],
				feed_dict={
					X_placeholder: X_test,
					Y_placeholder: Y_test,
					index_male_placeholder: index_male_test,
					index_female_placeholder: index_female_test,
					index_male_true_placeholder: index_male_true_test,
					index_male_false_placeholder: index_male_false_test,
					index_female_true_placeholder: index_female_true_test,
					index_female_false_placeholder: index_female_false_test,
					data_female_one_placeholder: test_data_one_female_prob,
					data_female_zero_placeholder: test_data_zero_female_prob,
					data_male_one_placeholder: test_data_one_male_prob,
					data_male_zero_placeholder: test_data_zero_male_prob
				}
	)
	print(f'total_test: {loss_total_test}, entropy_test: {loss_entropy_test}, accuracy_test: {accuracy_test}, imparity_test: {fairness_loss_test}')
	print("*****************")

	print('===train predictions===')
	print(pred_train)
	print(len(pred_train))
	print(sum(pred_train))
	train_female_one_prediction = np.where(pred_train[index_female_train] == 1)[0].astype(np.int32)
	train_female_zero_prediction = np.where(pred_train[index_female_train] == 0)[0].astype(np.int32)
	train_male_one_prediction = np.where(pred_train[index_male_train] == 1)[0].astype(np.int32)
	train_male_zero_prediction = np.where(pred_train[index_male_train] == 0)[0].astype(np.int32)
	print("females who were predicted a 1 label" + str(train_female_one_prediction.shape[0]/index_female_train.shape[0]))
	print("females who were predicted a 0 label" + str(train_female_zero_prediction.shape[0]/index_female_train.shape[0]))
	print("males who were predicted a 1 label" + str(train_male_one_prediction.shape[0]/index_male_train.shape[0]))
	print("males who were predicted a 0 label" + str(train_male_zero_prediction.shape[0]/index_male_train.shape[0]))

	print('===dev predictions===')
	print(pred_dev)
	print(len(pred_dev))
	print(sum(pred_dev))
	dev_female_one_prediction = np.where(pred_dev[index_female_dev] == 1)[0].astype(np.int32)
	dev_female_zero_prediction = np.where(pred_dev[index_female_dev] == 0)[0].astype(np.int32)
	dev_male_one_prediction = np.where(pred_dev[index_male_dev] == 1)[0].astype(np.int32)
	dev_male_zero_prediction = np.where(pred_dev[index_male_dev] == 0)[0].astype(np.int32)
	print("females who were predicted a 1 label" + str(dev_female_one_prediction.shape[0]/index_female_dev.shape[0]))
	print("females who were predicted a 0 label" + str(dev_female_zero_prediction.shape[0]/index_female_dev.shape[0]))
	print("males who were predicted a 1 label" + str(dev_male_one_prediction.shape[0]/index_male_dev.shape[0]))
	print("males who were predicted a 0 label" + str(dev_male_zero_prediction.shape[0]/index_male_dev.shape[0]))


	print('===test predictions===')
	print(pred_test)
	print(len(pred_test))
	print(sum(pred_test))
	test_female_one_prediction = np.where(pred_test[index_female_test] == 1)[0].astype(np.int32)
	test_female_zero_prediction = np.where(pred_test[index_female_test] == 0)[0].astype(np.int32)
	test_male_one_prediction = np.where(pred_test[index_male_test] == 1)[0].astype(np.int32)
	test_male_zero_prediction = np.where(pred_test[index_male_test] == 0)[0].astype(np.int32)
	print("females who were predicted a 1 label" + str(test_female_one_prediction.shape[0]/index_female_test.shape[0]))
	print("females who were predicted a 0 label" + str(test_female_zero_prediction.shape[0]/index_female_test.shape[0]))
	print("males who were predicted a 1 label" + str(test_male_one_prediction.shape[0]/index_male_test.shape[0]))
	print("males who were predicted a 0 label" + str(test_male_zero_prediction.shape[0]/index_male_test.shape[0]))

	print("************************************")
	print(train_data_one_female_prob)
	print(train_data_zero_female_prob)
	print(train_data_one_male_prob)
	print(train_data_zero_male_prob)
	print(dev_data_one_female_prob)
	print(dev_data_zero_female_prob)
	print(dev_data_one_male_prob)
	print(dev_data_zero_male_prob)
	print(test_data_one_female_prob)
	print(test_data_zero_female_prob)
	print(test_data_one_male_prob)
	print(test_data_zero_male_prob)

	pdb.set_trace()
	print('Pause before exit')
