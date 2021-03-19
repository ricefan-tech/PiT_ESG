import numpy as np
import matplotlib.pyplot as plt
from sklearn import utils
from tqdm.auto import tqdm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import datetime
from statsmodels.graphics.gofplots import qqplot_2samples
import io
import statsmodels.api as sm
import pdb
class CVAE(tf.keras.Model):
  

    def __init__(self, input_dim, n_latent, n_hidden=50, alpha=0.2, learning_rate=0.005):
        super(CVAE, self).__init__()
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.alpha = alpha
        #alpha gives weight to loss function parts: KL div and Reconstruction loss
        self.input_dim=input_dim
        self.learning_rate=learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.encoder=tf.keras.Sequential([tf.keras.layers.Flatten(),
                                          tf.keras.layers.Dense(units=self.n_hidden, activation=self.lrelu),
                                          tf.keras.layers.Dense(units=self.n_latent+self.n_latent, activation=self.lrelu, dtype=tf.float32)])
        #,tf.keras.layers.Reshape((-1,self.input_dim))
        self.decoder= tf.keras.Sequential([tf.keras.layers.Dense( units=self.n_hidden, activation=self.lrelu), 
                               tf.keras.layers.Dense(units=self.input_dim, activation=tf.nn.sigmoid)])
        
        self._log_dir = 'logs/scalars/' + datetime.datetime.now().strftime(
            "%d%m") + '/' + datetime.datetime.now().strftime("%H%M%S") + '/train'
        self._file_writer = tf.summary.create_file_writer(self._log_dir)
        self._file_writer.set_as_default()
        
        
    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))

    def encode(self, X, cond):
        X=tf.concat([X, cond], axis=1)
        mn, sd= tf.split(self.encoder(X), num_or_size_splits=2,axis=1)
        return mn, sd
        
        
    def reparameterize(self, mn, sd):
        epsilon = tf.random.normal(shape=tf.shape(mn))
        return mn + epsilon * tf.math.exp(sd)
    
    def decode(self, sample_z, cond):
        Y=tf.concat([sample_z, cond], axis=1)
        return self.decoder(Y)
    
    def kl_loss(self, mn, sd):
        #sum over dimension, ie. latent space dimension, of mn and sd, which are given as batchsize x latent space
        return -0.5 * tf.reduce_sum(1. + sd - tf.square(mn) - tf.exp(sd), 1)
    
    def recon_loss(self, x_pred, x_true):
        return tf.reduce_sum(tf.math.squared_difference(x_pred, x_true),1)
    
    def av_kl_loss(self,kl_loss):
        return tf.math.reduce_mean(kl_loss)
    
    def av_recon_loss(self, recon_loss):
        return tf.math.reduce_mean(recon_loss)
        

    def total_loss(self,x, cond):
        mn, sd=self.encode(x, cond)
        z=self.reparameterize(mn, sd)
        x_pred=self.decode(z,cond)
        recon_loss=self.recon_loss(x_pred, x)
        kl_loss=self.kl_loss(mn,sd)
    
        return tf.reduce_mean((1 - self.alpha) * recon_loss + self.alpha * kl_loss)
    
    @tf.function
    def train_step(self, x, cond):
        with tf.GradientTape() as tape:
            loss = self.total_loss(x, cond) 
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
    def plot_to_image(self,figure):
      
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close()
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image


    def QQ(self,samp, data):
        pp_x = sm.ProbPlot(data.T)
        pp_y = sm.ProbPlot(samp.T)
        fig=qqplot_2samples(pp_x,pp_y, xlabel="Quantiles of Data", ylabel="Quantiles of CVAR", line="45")
        fig.suptitle("CVAE vs. Data ")
        final_plot=self.plot_to_image(fig)
        return final_plot
    
    def fill_tboard(self, x,cond):
        mn, sd=self.encode(x,cond)
        z=self.reparameterize(mn,sd)
        x_pred=self.decode(z, cond)
        
        epoch=tf.cast(self.epoch, tf.int64)
        recon_loss=self.recon_loss(x_pred, x)
        #tf.reshape(recon_loss, shape=[])
        recon_loss=self.av_recon_loss(recon_loss)
        tf.summary.scalar('reconstr. loss', recon_loss, step=epoch)
        
        kl_loss=self.kl_loss(mn,sd)
        kl_loss=self.av_kl_loss(kl_loss)
        tf.summary.scalar('KL Div', kl_loss, step=epoch)
        
        ##################maybe QQ plot - need to think about qq of what and which conditional value 
        # gens=self.generate(cond, cond.shape[0])
        # qplot=self.QQ(gens, x)
        # tf.summary.image(qplot, "QQ Plot", step=n_epochs)
  
        

    @tf.function
    def train(self, data, data_cond, batch_size, train_size, n_epochs=10000):
        if len(data_cond.shape) == 1:
            data_cond = data_cond.reshape(-1, 1)
        data=tf.cast(data, dtype="float32")
        data_cond=tf.cast(data_cond, dtype="float32")
       
        trainsize=int(train_size*data.shape[0])
        testsize=data.shape[0]-trainsize
        x_train, x_test =tf.split(data, num_or_size_splits=[trainsize,testsize])
        y_train, y_test = tf.split(data_cond, num_or_size_splits=[trainsize,testsize])
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
        self.epoch=0
        for i in tf.range(1,n_epochs+1):
            self.epoch=i
          
            start_time=time.time()
            for x,y in train_dataset:
                self.train_step(x,y)
            end_time=time.time()
            t=end_time-start_time
         
            #print("Epoch " + str(i) + "Time "+ str(t))
            
            if tf.equal(i % 1000, 0):
                for x_test,y_test in test_dataset:
                    self.fill_tboard(x_test,y_test)
                tf.print("Epoch: ", i)
            #tf.cond(tf.equal(i % 1000, 0)  ,lambda: self.fill_tboard(),lambda: self.false_fn())   
    def generate(self, cond, n_samples):
        randoms = tf.random.normal(shape=(n_samples, self.n_latent))
        ######################################conditional distribution on one conditional value
        cond=tf.repeat(cond, repeats=[n_samples], axis=0)
        
        cond=tf.cast(cond, dtype="float32")
        cond=tf.reshape(cond, shape=(-1,1))
       
        samples = self.decode(randoms,cond)

        return samples

