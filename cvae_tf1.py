import numpy as np
import matplotlib.pyplot as plt
from sklearn import utils
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class CVAE(object):
    """Conditional Variational Auto Encoder (CVAE)."""

    def __init__(self, scaler,rf, recon=0.5, kl=0.5, n_latent=5, n_hidden=50, alpha=0.5):
        #pass data, cond as already built dataset
        self.recon=recon
        self.kl=kl
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.rf=rf
        self.freq="W"
        self.scaler=scaler
        
    
    def process_data(self, data):
        w=[]
        for i in range(data.shape[0]):
    
            tmp=self.scaler.transform(data[i])
            if tmp.max()>1:
                print(i)
            w.append(tmp)    
        a=np.array([we for we in w])
        return np.array([val.T.reshape(self.rf*val.shape[0],) for val in a], dtype=object)
    
        
    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))

    def encoder(self, X_in, cond, input_dim):
        with tf.variable_scope("encoder", reuse=None):
            x = tf.concat([X_in, cond], axis=1)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=self.n_hidden, activation=self.lrelu)
            x = tf.layers.dense(x, units=15, activation=self.lrelu)
            #x = tf.layers.dense(x, units=30, activation=self.lrelu)
            #x = tf.layers.dense(x, units=30, activation=self.lrelu)
            mn = tf.layers.dense(x, units=self.n_latent, activation=self.lrelu)
            sd = tf.layers.dense(x, units=self.n_latent, activation=self.lrelu)
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent]))
            z = mn + tf.multiply(epsilon, tf.exp(sd / 2.))

            return z, mn, sd
    
    def decoder(self, sampled_z, cond, input_dim):
        with tf.variable_scope("decoder", reuse=None):
            x = tf.concat([sampled_z, cond], axis=1)
            x = tf.layers.dense(x, units=self.n_hidden, activation=self.lrelu)
            x = tf.layers.dense(x, units=15, activation=self.lrelu)
            #x = tf.layers.dense(x, units=30, activation=self.lrelu)
            #x = tf.layers.dense(x, units=70, activation=self.lrelu)
            #x = tf.layers.dense(x, units=self.n_hidden, activation=self.lrelu)
            x = tf.layers.dense(x, units=input_dim, activation=tf.nn.sigmoid)
            x = tf.reshape(x, shape=[-1, input_dim])

            return x
        
   
    def train(self, data, data_cond, n_epochs=10000, learning_rate=0.005,
              show_progress=True):
        print("Train")
        
      
        data = utils.as_float_array(data)
        data_cond = utils.as_float_array(data_cond)
        if len(data_cond.shape) == 1:
            data_cond = data_cond.reshape(-1, 1)
        print(data.max())
        assert data.max() <= 1. and data.min() >=0., \
            "All features of the dataset must be between 0 and 1."
        
        tf.reset_default_graph()

        input_dim = data.shape[1]
        dim_cond = data_cond.shape[1]

        X_in = tf.placeholder(dtype=tf.float32, shape=[None, input_dim],
                              name="X")
                            
        self.cond = tf.placeholder(dtype=tf.float32, shape=[None, dim_cond],
                                   name="c")
        Y = tf.placeholder(dtype=tf.float32, shape=[None, input_dim],
                           name="Y")

        Y_flat = Y
        
       
        self.sampled, mn, sd = self.encoder(X_in, self.cond, input_dim=input_dim)
        self.dec = self.decoder(self.sampled, self.cond, input_dim=input_dim)

        unreshaped = tf.reshape(self.dec, [-1, input_dim])
        decoded_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
        latent_loss = -0.5 * tf.reduce_sum(1. + sd - tf.square(mn) - tf.exp(sd), 1)
        self.d=tf.reduce_mean(decoded_loss)
        self.l=tf.reduce_mean(latent_loss)
        self.loss = tf.reduce_mean(self.recon *10930* decoded_loss + self.kl * latent_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        for i in range(n_epochs):
            
            #passing Y for reconstruction loss
            self.sess.run(optimizer, feed_dict={X_in: data, self.cond: data_cond, Y: data})
        
            if i % 2000 ==0:
                self.kl=i/10000
            if i % 1000 ==0:
                
                ls, dec, lat, d = self.sess.run([self.loss, self.d,self.l, self.dec], feed_dict={X_in: data, self.cond: data_cond, Y: data})
                
                # projections = np.random.randint(0, data.shape[1], size=2)
                # plt.figure()
                # plt.scatter(data[:, projections[0]], data[:, projections[1]])
                # plt.scatter(d[:, projections[0]], d[:, projections[1]])
                # plt.xlabel(str(projections[0]))
                # plt.xlabel(str(projections[1]))
                # plt.show()
                
                print(i, self.kl, ls, dec, lat)
    
    def generate(self, cond, n_samples=None):
        cond = utils.as_float_array(cond)

        if n_samples is not None:
            randoms = np.random.normal(0, 1, size=(n_samples, self.n_latent))
            cond = np.repeat(cond, n_samples, axis=0)
        else:
            randoms = np.random.normal(0, 1, size=(1, self.n_latent))
            cond = cond

        samples = self.sess.run(self.dec, feed_dict={self.sampled: randoms, self.cond: cond})
       
        if n_samples is None:
            print(samples)
            #inverse trafo takes in shape n_features, n_samples
            #return self.scaler.inverse_transform(np.concatenate([samples[0].reshape(5,-1).T, cond.reshape(-1,1)], axis=1))
            #return self.scaler.inverse_transform(np.concatenate([samples[0].reshape(2,-1).T, np.repeat(cond,5, axis=1).reshape(-1,1)], axis=1))
      
        # s=samples[:,:5].reshape(-1,1)
        # v=samples[:,5:10].reshape(-1,1)
        # f=samples[:,10:].reshape(-1,1)
        
        # return self.scaler.inverse_transform(np.concatenate([s,v,f], axis=1))
        return samples
        #return self.scaler.inverse_transform(samples)
        
        
    def sim(self, init_vix,weeks, path):
       
        sims_cvae=np.zeros((weeks*5,path))
       
        #initial VIX value for simulation start
        print("initial vix ", init_vix)
        for i in range(path):
            #these are scaled values for vix indicator \pm 1 (sclaer makes problems else)
            cond=init_vix
            print(cond)
            for j in range(weeks):
                s= j * 5
                e = (j + 1) * 5
                #sc_cond=self.scaler.inverse_transform(np.asarray([1,1,cond]).reshape(1,-1))
                if np.any(cond> 40):
                    cond=np.where(cond>40, 40, cond)
                elif np.any(cond<10):
                    cond=np.where(cond>40, 40, cond)
                o=np.ones((1,2))
                tmp=self.scaler.transform(np.concatenate([o,cond.reshape(-1,1)], axis=1))
              
                #cond=tmp[0][2]
                cond=tmp[:,2].reshape(1,-1)
                cond=self.generate(cond)
                #print(cond)
                sims_cvae[s:e,i]=cond[:,0]
                cond=np.around(cond[-1,1])
                
        #     sims_cvae[:,:2,i]=np.exp(sims_cvae[:,:2,i].cumsum(axis=0).astype(np.float64))   
           
        # sims_cvae[:,:,:]=np.r_[a,sims_cvae[:-1,:,:]]
        # startspx=datas_np[start*5,0]
        # startvix=datas_np[start*5,1]
        # sims_cvae[:,0,:]*=startspx
        # sims_cvae[:,1,:]*=startvix
            
        return sims_cvae
    