#%%
import os
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from tensorflow.keras import regularizers

# Load data from source domain
# User embeddings
s_u_list = pd.read_csv('../Dataset/Book/user_embedding/user_list.csv')
s_u = np.genfromtxt('../Dataset/Book/user_embedding/user_emb.csv', delimiter=',', dtype=np.float32)

# Item embeddings
s_i_list = pd.read_csv('../Dataset/Book/item_embedding/item_list.csv').set_index('item_id')
s_i_emb = np.genfromtxt('../Dataset/Book/item_embedding/item_emb.csv', delimiter=',', dtype=np.float32)

s_i = []
s_i_unexp = []
s_i_rel = []

# Construct positive and negative sample pairs
for i in range(len(s_u_list)):
  u_temp = s_u_list.iloc[i]['user_id']

  u_unexp_df = pd.read_csv('../Dataset/Book/user_unexpectedness_samples_movie/' + u_temp + '.csv', index_col=0)[
      'item_id'].values
  # convert id to item embeddings
  for id in u_unexp_df:
      i_ix = s_i_list.loc[id]['item_index']
      s_i_unexp.append(s_i_emb[i_ix])

  u_rel_df = pd.read_csv('../Dataset/Book/user_relevance_samples_movie/' + u_temp + '.csv', index_col=0)[
      'item_id'].values
  # convert id to item embeddings
  for id in u_rel_df:
      i_ix = s_i_list.loc[id]['item_index']
      s_i_rel.append(s_i_emb[i_ix])


  u_temp_df = pd.read_csv('../Dataset/Book/user_pairs_book/' + u_temp + '.csv')
  s_i_pos = u_temp_df[u_temp_df['serendipity label'] == 1]['item_id'].values
  s_i_neg = u_temp_df[u_temp_df['serendipity label'] == 0]['item_id'].values
  s_i_all = np.concatenate((s_i_pos, s_i_neg), axis=0)

  for id in s_i_all:
    i_ix = s_i_list.loc[id]['item_index']
    s_i.append(s_i_emb[i_ix])

# Reshape data
s_i = np.array(s_i).reshape([-1, 100, 128])
s_i_unexp = np.array(s_i_unexp).reshape([-1, 100, 128])
s_i_rel = np.array(s_i_rel).reshape([-1, 100, 128])

# Load data from target domain
# User embeddings
t_u_list = pd.read_csv('../Dataset/Movie/user_embedding/user_list.csv')
t_u = np.genfromtxt('../Dataset/Movie/user_embedding/user_emb.csv', delimiter=',', dtype=np.float32)

# Item embeddings
t_i_list = pd.read_csv('../Dataset/Movie/item_embedding/item_list.csv').set_index('item_id')
t_i_emb = np.genfromtxt('../Dataset/Movie/item_embedding/item_emb.csv', delimiter=',', dtype=np.float32)

t_i = []
t_i_unexp = []
t_i_rel = []

# Construct positive and negative sample pairs

for i in range(len(t_u_list)):
  u_temp = t_u_list.iloc[i]['user_id']
  u_unexp_df = pd.read_csv('../Dataset/Movie/user_unexpectedness_samples_movie/' + u_temp + '.csv', index_col=0)['item_id'].values
  # convert id to item embeddings
  for id in u_unexp_df:
    i_ix = t_i_list.loc[id]['item_index']
    t_i_unexp.append(t_i_emb[i_ix])

  u_rel_df = pd.read_csv('../Dataset/Movie/user_relevance_samples_movie/' + u_temp + '.csv', index_col=0)['item_id'].values
  # convert id to item embeddings
  for id in u_rel_df:
      i_ix = t_i_list.loc[id]['item_index']
      t_i_rel.append(t_i_emb[i_ix])


  u_temp_df = pd.read_csv('../Dataset/Movie/user_pairs_movie/' + u_temp + '.csv')
  t_i_pos = u_temp_df[u_temp_df['serendipity label'] == 1]['item_id'].values # positive samples
  t_i_neg = u_temp_df[u_temp_df['serendipity label'] == 0]['item_id'].values # negative samples
  t_i_all = np.concatenate((t_i_pos, t_i_neg), axis=0)
  # convert id to item embeddings
  for id in t_i_all:
    i_ix = t_i_list.loc[id]['item_index']
    t_i.append(t_i_emb[i_ix])



t_i = np.array(t_i).reshape([-1, 100, 128])
t_i_unexp = np.array(t_i_unexp).reshape([-1, 100, 128])
t_i_rel = np.array(t_i_rel).reshape([-1, 100, 128])

# Hyperparameters of SerenCDR
grp_u = 64 # user groups
grp_i = 64  # item groups
dim = 64 # serendipity features

# Domain-specific knowledge learning module for source domain
class sd_specific(layers.Layer):
  def __init__(self):
    super().__init__()
    dim = 128
    # for user
    self.d1 = layers.Dense(dim, activation='relu')
    # for item
    self.d2 = layers.Dense(dim, activation='relu')

  def call(self, u, i):
    u_sp = self.d1(u)
    i_sp = self.d2(i)

    return u_sp, i_sp

# Domain-specific knowledge learning module for target domain
class td_specific(layers.Layer):
  def __init__(self):
    super().__init__()
    # for user
    self.d1 = layers.Dense(dim, activation='relu')
    # for item
    self.d2 = layers.Dense(dim, activation='relu')

  def call(self, u, i):
    u_sp = self.d1(u)
    i_sp = self.d2(i)

    return u_sp, i_sp

# Domain-sharing knowledge learning module
# User
class sharing_u(layers.Layer):
  def __init__(self):
    super().__init__()
    # for source
    self.d1 = layers.Dense(grp_u, activation='relu')
    # for target
    self.d2 = layers.Dense(grp_u, activation='relu')
    # sharing knowledge
    self.us = self.add_weight("us",shape=(grp_u, dim), regularizer=tf.keras.regularizers.l2())
    # linear projection
    self.t = self.add_weight("t",shape=(dim, dim), regularizer=tf.keras.regularizers.l2())

  def call(self, s_u, t_u):
    p_s = self.d1(s_u)
    p_t = self.d2(t_u)

    us_sh = tf.matmul(p_s, self.us)
    ut = tf.matmul(self.us, self.t)
    ut_sh = tf.matmul(p_t, ut)
    # ut_sh = tf.matmul(p_t, self.us)

    return us_sh, ut_sh

# Item
class sharing_i(layers.Layer):
  def __init__(self):
    super().__init__()
    # for source
    self.d1 = layers.Dense(grp_i, activation='relu')
    # for target
    self.d2 = layers.Dense(grp_i, activation='relu')
    # sharing knowledge
    self.vs = self.add_weight("vs",shape=(grp_i, dim), regularizer=tf.keras.regularizers.l2())
    # linear projection
    self.t = self.add_weight("t",shape=(dim, dim), regularizer=tf.keras.regularizers.l2())

  def call(self, s_i, t_i):
    q_s = self.d1(s_i)
    q_t = self.d2(t_i)

    is_sh = tf.matmul(q_s, self.vs)
    it = tf.matmul(self.vs, self.t)
    it_sh = tf.matmul(q_t, it)
    # it_sh = tf.matmul(q_t, self.vs)

    return is_sh, it_sh

# Main function
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    # dim = 128  # serendipity features

  def call(self, s_u, s_i, s_i_unexp, s_i_rel, t_u, t_i, t_i_unexp, t_i_rel,):
    # domain-specific
    # source domain
    sd_sp_u, sd_sp_i = sd_specific()(s_u, s_i)
    # target domain
    td_sp_u, td_sp_i = td_specific()(t_u, t_i)
    # domain-sharing
    # users
    sd_sh_u, td_sh_u = sharing_u()(s_u, t_u)
    # items
    sd_sh_i, td_sh_i  = sharing_i()(s_i, t_i)

    # Concatenating - fusion for source domain
    sd_sh_u = tf.reshape(sd_sh_u,[-1,1,sd_sh_u.shape[-1]])
    sd_sp_u = tf.reshape(sd_sp_u,[-1,1,sd_sp_u.shape[-1]])
    sd_u = tf.keras.layers.Concatenate(axis=1)([sd_sh_u, sd_sp_u]) # user
    sd_i = tf.keras.layers.Concatenate(axis=1)([sd_sh_i, sd_sp_i]) # item

    # Concatenating - fusion for target domain
    td_sh_u = tf.reshape(td_sh_u,[-1,1,td_sh_u.shape[-1]])
    td_sp_u = tf.reshape(td_sp_u,[-1,1,td_sp_u.shape[-1]])
    td_u = tf.keras.layers.Concatenate(axis=1)([td_sh_u, td_sp_u]) # user
    td_i = tf.keras.layers.Concatenate(axis=1)([td_sh_i, td_sp_i]) # item




    # prediction - source domain
    r_s = tf.matmul(sd_u, sd_i, transpose_b=True)
    r_s = tf.reshape(r_s,[-1,r_s.shape[-1]])
    r_s = tf.keras.activations.sigmoid(r_s)
    #r_s = tf.nn.softmax(r_s)

    # pairs - source domain
    rs_pos = r_s[:,:50]
    rs_neg = r_s[:,50:]
    pair_rs = rs_pos-rs_neg

    # prediction - target domain
    r_t = tf.matmul(td_u, td_i, transpose_b=True)
    r_t = tf.reshape(r_t, [-1, r_t.shape[-1]])
    r_t = tf.keras.activations.sigmoid(r_t)

    # pairs - target domain
    rt_pos = r_t[:,:50]
    rt_neg = r_t[:,50:]
    pair_rt = rt_pos-rt_neg

    # auxiliary - source domain
    rs_unexp = tf.matmul(sd_u, s_i_unexp, transpose_b=True)
    rs_unexp = tf.reshape(rs_unexp, [-1, rs_unexp.shape[-1]])
    rs_unexp = tf.keras.activations.sigmoid(rs_unexp)

    rs_rel = tf.matmul(sd_u, s_i_rel, transpose_b=True)
    rs_rel = tf.reshape(rs_rel, [-1, rs_rel.shape[-1]])
    rs_rel = tf.keras.activations.sigmoid(rs_rel)

    aux_s = tf.negative(tf.math.log(tf.math.divide(rs_rel, rs_unexp)))

    # auxiliary - target domain
    rt_unexp = tf.matmul(td_u, t_i_unexp, transpose_b=True)
    rt_unexp = tf.reshape(rt_unexp, [-1, rs_unexp.shape[-1]])
    rt_unexp = tf.keras.activations.sigmoid(rt_unexp)

    rt_rel = tf.matmul(td_u, t_i_rel, transpose_b=True)
    rt_rel = tf.reshape(rt_rel, [-1, rt_rel.shape[-1]])
    rt_rel = tf.keras.activations.sigmoid(rt_rel)

    aux_t = tf.negative(tf.math.log(tf.math.divide(rt_rel, rt_unexp)))

    return r_s, pair_rs, aux_s, r_t, pair_rt, aux_t

# Create an instance of the model
model = MyModel()

# Optimizer
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Loss
train_loss = tf.keras.metrics.Mean(name='train_loss')

# Evaluation metrics
def hit_ratio(predictions, k):
    topk_list = np.argsort(predictions, axis=1)[:,:k]
    n = 0
    for i in range(len(topk_list)):
        if np.sort(topk_list[i])[0] == 0:
            n = n + 1
    hr = n/len(topk_list)

    return hr

def ndcg(sre_labels, predictions, k):
    topk_list = np.argsort(predictions, axis=1)[:,:k]
    pred_m = np.zeros((predictions.shape[0],predictions.shape[1]))

    for i in range(len(topk_list)):
        pred_m[i][topk_list[i]] = 1


    ndcg = ndcg_score(sre_labels, pred_m, k=k)

    return ndcg

# Training
def train_step(s_u_train, s_i_train, s_i_unexp_train, s_i_rel_train, sd_train_labels, t_u_train, t_i_train, t_i_unexp_train, t_i_rel_train, td_train_labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    sd_predictions, sd_pair_pre, sd_aux, td_predictions, td_pair_pre, td_aux = model(s_u_train, s_i_train, s_i_unexp_train, s_i_rel_train, t_u_train, t_i_train, t_i_unexp_train, t_i_rel_train, training=True)

    loss_source = loss_object(sd_train_labels[:,:50], sd_pair_pre) # Loss of source domain
    loss_target = loss_object(td_train_labels[:,:50], td_pair_pre) # Loss of target domain
    loss_pair = loss_source + loss_target

    loss_source_aux = sd_aux
    loss_target_aux = td_aux
    loss_auxiliary = loss_source_aux + loss_target_aux
    loss = loss_pair + loss_auxiliary
    # print(loss)

  gradients = tape.gradient(loss_pair, model.trainable_variables)
  # print(gradients)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)

# Testing
from sklearn.metrics import ndcg_score

def test_step(s_u_test, s_i_test, sd_test_labels, t_u_test, t_i_test, td_test_labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  sd_predictions, sd_pair_pre, td_predictions, td_pair_pre = model(s_u_test, s_i_test, t_u_test, t_i_test, training=False)

  # Hit Ratio
  hr_1 = hit_ratio(td_predictions, 1)
  hr_5 = hit_ratio(td_predictions, 5)
  hr_10 = hit_ratio(td_predictions, 10)

  # NDCG
  ndcg_5 = ndcg(td_test_labels, td_predictions, 5)
  ndcg_10 = ndcg(td_test_labels, td_predictions, 10)

  return hr_1, hr_5, hr_10, ndcg_5, ndcg_10, td_predictions.numpy()

# Start
def get_batch_data(u_l, tu_l, t_u, t_i, t_i_unexp, t_i_rel):
  tu_l = tu_l.set_index('user_id')
  tu = []
  ti = []
  ti_unexp = []
  ti_rel = []
  for i in range(len(u_l)):
    u_temp = u_l[i].replace('.csv','')
    u_ix = tu_l.loc[u_temp]
    tu.append(t_u[u_ix])
    ti.append(t_i[u_ix][0])
    ti_unexp.append(t_i_unexp[u_ix][0])
    ti_rel.append(t_i_rel[u_ix][0])
  tu = np.array(tu)
  ti = np.array(ti)
  ti_unexp = np.array(ti_unexp)
  ti_rel = np.array(ti_rel)
  return tu, ti, ti_unexp, ti_rel

# Labels for source domain
sd_train_labels = np.ones((s_i.shape[0],50))
sd_test_labels = np.zeros((s_i.shape[0],s_i.shape[1]))
sd_test_labels[:,0] = 1

# Labels for target domain
# Training
td_train_labels = np.ones((496,50))
# Testing
td_test_labels = np.zeros((123,100))
td_test_labels[:,0] = 1


batch_size = 32
file_path = '../Dataset/Movie/5_fold_data/'
EPOCHS = 100

for f in range(5):
  print('----fold ' + str(f) + '----')
  u_train = os.listdir(file_path + 'fold_' + str(f) + '/train/')
  u_test = os.listdir(file_path + 'fold_' + str(f) + '/test/')

  t_u_train, t_i_train, t_i_unexp_train, t_i_rel_train = get_batch_data(u_train, t_u_list, t_u, t_i, t_i_unexp, t_i_rel)

  train_td = tf.data.Dataset.from_tensor_slices((t_u_train, t_i_train, t_i_unexp_train, t_i_rel_train, td_train_labels)).batch(batch_size)
  train_sd = tf.data.Dataset.from_tensor_slices((s_u, s_i, s_i_unexp, s_i_rel, sd_train_labels)).batch(5*batch_size)

  t_u_test, t_i_test, t_i_unexp_test, t_i_rel_test = get_batch_data(u_test, t_u_list, t_u, t_i, t_i_unexp, t_i_rel)


  model = MyModel()
  result = np.zeros((0, 5))

  for epoch in range(EPOCHS):
      # Reset the metrics at the start of the next epoch
      train_loss.reset_states()
      pred = np.zeros((1,100))

      for (s_u, s_i, s_i_unexp, s_i_rel, sd_train_labels), (t_u_train, t_i_train, t_i_unexp_train, t_i_rel_train, td_train_labels) in zip(train_sd, train_td):
          train_step(s_u, s_i, s_i_unexp, s_i_rel, sd_train_labels, t_u_train, t_i_train, t_i_unexp_train, t_i_rel_train, td_train_labels)

      print(
        'Epoch %d --' % (epoch + 1),
        'Loss: %.6f,' % train_loss.result(),
      )

  hr_1, hr_5, hr_10, ndcg_5, ndcg_10, pred = test_step(s_u, s_i, s_i_unexp, s_i_rel, sd_test_labels, t_u_test, t_i_test, t_i_unexp_test, t_i_rel_test, td_test_labels)


  r = np.array([[hr_1, hr_5, hr_10,  ndcg_5, ndcg_10]])

  result = np.concatenate((result, r), axis=0)

  print(
          'Fold %d --' % (f + 1),
          'Test HR@1: %.4f,' % hr_1,
          'Test HR@5: %.4f,' % hr_5,
          'Test HR@10: %.4f,' % hr_10,
          'NDCG@5: %.4f,' % ndcg_5,
          'NDCG@10: %.4f' % ndcg_10,
  )

  col = ['HR@1', 'HR@5', 'HR@10', 'NDCG@5', 'NDCG@10']
  df = pd.DataFrame(data=result, columns=col)
  df.to_csv('../Results/SerenCDR/CrossSeren_fold_%d.csv' % (f+1))
