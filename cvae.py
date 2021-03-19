import numpy as np
import matplotlib.pyplot as plt
from sklearn import utils
from tqdm.auto import tqdm
import tensorflow as tf
import datetime
import os

class CVAE(object):
    """Conditional Variational Auto Encoder (CVAE)."""

    def __init__(self, n_latent,kl=0, n_hidden=50, alpha=0.2):
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.kl=kl
        self._model_folder='CVAE/models/' + datetime.datetime.now().strftime("%d%m")
    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))

    def encoder(self, X_in, cond, input_dim, reuse=None):
        with tf.variable_scope("encoder", reuse=reuse):
            x = tf.concat([X_in, cond], axis=1)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=self.n_hidden, activation=self.lrelu)
            #x = tf.layers.dense(x, units=20, activation=self.lrelu)
            # x = tf.layers.dense(x, units=120, activation=self.lrelu)
            # x = tf.layers.dense(x, units=80, activation=self.lrelu)
            # x = tf.layers.dense(x, units=45, activation=self.lrelu)
            # x = tf.layers.dense(x, units=18, activation=self.lrelu)
            mn = tf.layers.dense(x, units=self.n_latent, activation=self.lrelu)
            sd = tf.layers.dense(x, units=self.n_latent, activation=self.lrelu)
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent]))
            z = mn + tf.multiply(epsilon, tf.exp(sd / 2.))

            return z, mn, sd
    
    def decoder(self, sampled_z, cond, input_dim, reuse=None):
        with tf.variable_scope("decoder", reuse=reuse):
            x = tf.concat([sampled_z, cond], axis=1)
            
            x = tf.layers.dense(x, units=self.n_hidden, activation=self.lrelu)
            #x = tf.layers.dense(x, units=20, activation=self.lrelu)
            # x = tf.layers.dense(x, units=120, activation=self.lrelu)
            # x = tf.layers.dense(x, units=80, activation=self.lrelu)
            # x = tf.layers.dense(x, units=45, activation=self.lrelu)
            # x = tf.layers.dense(x, units=18, activation=self.lrelu)
            x = tf.layers.dense(x, units=input_dim, activation=tf.nn.sigmoid)
            x = tf.reshape(x, shape=[-1, input_dim])

            return x
    
    def model_from_saved(self, model_path, meta_path):
        
        tf.reset_default_graph()
        #latest_ckp = tf.train.latest_checkpoint('./') 
        #print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')
        with tf.Session() as sess:
            saver=tf.train.import_meta_graph(meta_path)
            saver.restore(sess, tf.train.latest_checkpoint(model_path))

    def save_model(self, saver, epochs):
        if not os.path.exists(self._model_folder):
            os.makedirs(self._model_folder)
        path=self._model_folder+ '/' + datetime.datetime.now().strftime("%d%m-%H%M%S") + str(epochs)
        saver.save(self.sess, path)
        print("model saved in: "+path)
        
    def train(self, scaler_data, scaler_cond, data, data_cond, n_epochs=10000, learning_rate=0.005, show_progress=True):

        data = utils.as_float_array(data)
        data_cond = utils.as_float_array(data_cond)

        if len(data_cond.shape) == 1:
            data_cond = data_cond.reshape(-1, 1)

        assert data.max() <= 1. and data.min() >=0., \
            "All features of the dataset must be between 0 and 1."

        tf.reset_default_graph()
       
        input_dim = data.shape[1]
        dim_cond = data_cond.shape[1]

        X_in = tf.placeholder(dtype=tf.float32, shape=[None, input_dim],
                              name="X")
        self.scaler_data=scaler_data
        self.scaler_cond=scaler_cond
        self.cond = tf.placeholder(dtype=tf.float32, shape=[None, dim_cond],
                                   name="c")
        Y = tf.placeholder(dtype=tf.float32, shape=[None, input_dim],
                           name="Y")

        Y_flat = Y

        self.sampled, mn, sd = self.encoder(X_in, self.cond,input_dim)
        self.dec = self.decoder(self.sampled, self.cond,input_dim)
        
        
        #unreshaped = tf.reshape(self.dec(self.sampled, self.cond,input_dim), [-1, input_dim])
        decoded_loss = tf.reduce_sum(tf.squared_difference(self.dec, Y_flat), 1)
        latent_loss = -0.5 * tf.reduce_sum(1. + sd - tf.square(mn) - tf.exp(sd), 1)

        self.loss = tf.reduce_mean((1 - self.alpha) * decoded_loss + self.alpha * latent_loss)
        self.decloss=tf.reduce_mean(decoded_loss)
        self.latloss=tf.reduce_mean(latent_loss)
        
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #print(tf.get_collection((tf.GraphKeys.GLOBAL_VARIABLES)))
        #vars_to_save=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="encoder")+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="decoder")
        saver=tf.train.Saver()
        # for i, var in enumerate(saver._var_list):   
        #     print('Var {}: {}'.format(i, var))
        for i in range(n_epochs):
            self.sess.run(optimizer, feed_dict={X_in: data, self.cond: data_cond, Y: data})
         
            if not i % 2000 and show_progress:
                ls, d, dec, lat = self.sess.run([self.loss, self.dec, self. decloss, self.latloss], feed_dict={X_in: data, self.cond: data_cond, Y: data})

                # projections = np.random.randint(0, data.shape[1], size=2)

                # plt.scatter(data[:, projections[0]], data[:, projections[1]])
                # plt.scatter(d[:, projections[0]], d[:, projections[1]])
                # plt.show()
                
                print(i, ls, dec, lat)
                
                samps=self.generate(data_cond[0])
                recon=np.mean((data-samps)**2)
                print('conditional sampling: ',recon)
        self.save_model(saver, n_epochs)
       
        
    def generate(self, cond, n_samples=None):
        cond = utils.as_float_array(cond)

        if n_samples is not None:
            randoms = np.random.normal(0, 1, size=(n_samples, self.n_latent))
            cond = [list(cond)] * n_samples
        else:
            randoms = np.random.normal(0, 1, size=(1, self.n_latent))
            cond = [list(cond)]

        samples = self.sess.run(self.dec, feed_dict={self.sampled: randoms, self.cond: cond})

        if n_samples is None:
            return samples[0]

        return samples

