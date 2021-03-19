import datetime
import numpy as np
import pandas_datareader as pdr
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import utils
from utils.leadlag import leadlag
from cvae import CVAE
import datetime
import os
import joblib
import tensorflow as tf

#tf.enable_eager_execution()
class MarketGenerator:
    def __init__(self, ticker,ticker2,nhidden, nlatent, alpha, start=datetime.date(2000, 1, 3),
                 end=datetime.date(2020, 12, 18), freq="M",
                 sig_order=4, saved_model=None):

        self.ticker = ticker
        self.ticker2=ticker2
        self.start = start
        self.end = end
        self.freq = freq
        self.order = sig_order
       
       
       
        if saved_model is None:
             self._load_data()
             self._scaler_folder='CVAE/scalers/' + datetime.datetime.now().strftime("%d%m")
             self._build_dataset()
             self.generator = CVAE(n_hidden=nhidden, n_latent=nlatent, alpha=alpha)
        else:
            self.generator=saved_model
          


    def backfilling(self,data, logs=True):
        l=pd.date_range(data.index[0], data.index[-1], freq="B")
        dind=data.index
        new=pd.DataFrame(index=l, columns=["data"])
        if logs:
            #backfill logs
            for i in range(len(l)):
                if l[i] in dind:
                    new.loc[l[i],:]=data.loc[l[i]]
                else: 
                    new.loc[l[i],:]=np.zeros((1,2))
        else:
            #backfill levels
            for i in range(len(l)):
                if l[i] in dind:
                    new.loc[l[i],:]=data.loc[l[i]]
                else: 
                    new.loc[l[i],:]=new.loc[l[i-1]]
                    
              
        return new
    def _load_data(self):
        try:
            self.data = pdr.get_data_yahoo(self.ticker, self.start, self.end)["Close"]
            self.data=self.backfilling(self.data, logs=False)
            self.data.data=np.log((self.data.data /self.data.data.shift(1)).astype(np.float64))
            self.cond=pdr.get_data_yahoo(self.ticker2, self.start, self.end)["Close"]
            self.cond=self.backfilling(self.cond, logs=False)
            # self.cond.where(self.cond>=10.0, 10.0, inplace=True)
            # self.cond.where(self.cond<=40.0, 40.0, inplace=True)
            #self.data=pd.concat([self.data, cond], axis=1)
        except:
            raise RuntimeError(f"Could not download data for {self.ticker} from {self.start} to {self.end}.")

        self.windows = []
        for _, window in self.data.resample(self.freq):
            values = window.values# / window.values[0]
            path = leadlag(values)
            #we loose first nan value due to leadlag trafo 
            self.windows.append(path)
        self.windows2 = []
        for _, window in self.cond.resample(self.freq):
            values = window.values# / window.values[0]
            path = leadlag(values)

            self.windows2.append(path)
        
    def _save_scaler(self,scaler1,scaler2):
        if not os.path.exists(self._scaler_folder):
            os.makedirs(self._scaler_folder)
        path=self._scaler_folder+ "/" + datetime.datetime.now().strftime("%d%m-%H%M%S") 
        joblib.dump(scaler1, path+"_scaler_data.save")
        joblib.dump(scaler2, path+"_scaler_cond.save")
        print("scalers saved in: "+path)
    
    def _restore_scaler(self, scaler_path1, scaler_path2):
        self.scaler=joblib.load(scaler_path1)
        self.scaler2=joblib.load(scaler_path2)
        
    def _build_dataset(self):
        if self.order:
            self.orig_logsig = np.array([self._logsig(path) for path in tqdm(self.windows, desc="Computing log-signatures")])
        else:
            self.orig_logsig = np.array([path[::2, 1] for path in self.windows])
            self.orig_logsig = np.array([p for p in self.orig_logsig if len(p) >= 5])
            ###everything for the conditions
            self.c = np.array([path[::2, 1] for path in self.windows2])
            self.c = np.array([p for p in self.c if len(p) >= 5]) 
        self.scaler = MinMaxScaler(feature_range=(0.00001, 0.99999))
        logsig = self.scaler.fit_transform(self.orig_logsig)
        self.scaler2 = MinMaxScaler(feature_range=(0.00001, 0.99999))
    
        #cut off and transform for integer conditioning values
        rounded_c=np.around(self.c.astype(np.float64))
        rounded_c=np.where(rounded_c<=10.0, 10.0, rounded_c)
        rounded_c=np.where(rounded_c>=40.0, 40.0, rounded_c)
        self.rounded_cond=self.scaler2.fit_transform(rounded_c)
        self.logsigs=logsig[1:]
        self.conditions = self.rounded_cond[:-1,-1]
        self._save_scaler(self.scaler, self.scaler2)
        
        
        
    def train(self, lrate, n_epochs=10000, show=True):
        self.generator.train(self.scaler, self.scaler2, self.logsigs, self.conditions, learning_rate=lrate, n_epochs=n_epochs, show_progress=show)

    def generate_from_saved(self,cond, n_samples, inpt_dim):
        cond = utils.as_float_array(cond).reshape(-1,1)
            
        if n_samples is not None:
            randoms = np.random.normal(0, 1, size=(n_samples, self.generator.n_latent))
            cond = [list(cond)] * n_samples
        else:
            randoms = np.random.normal(0, 1, size=(1, self.generator.n_latent))
            
            #cond = [list(cond)]
        samples=self.generator.decoder(randoms, cond.reshape(1,1), inpt_dim, reuse=tf.AUTO_REUSE)            
        
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        if n_samples is None:
            #generated= samples[0].eval()
            generated=samples[0]
        else:
            generated= samples.eval()
        return generated
    
    def generate(self, cond,n_samples=None, normalised=True, saved_model=True, inpt_dim=5):
        if saved_model is None:
            generated = self.generator.generate(cond, n_samples=n_samples)
        else:
           
            generated=self.generate_from_saved(cond, n_samples, inpt_dim=inpt_dim)
            
        if normalised:
            return generated
        
        if n_samples is None:
            return self.scaler.inverse_transform(generated.reshape(1,-1))[0]

        return self.scaler.inverse_transform(generated)

    
    def sim(self, cond, weeks, path, n_samples=None, saved_model=None, scaler_path1=None, scaler_path2=None, inpt_dim=5, normalised=True):
        sim_cvae=np.zeros((weeks,inpt_dim, path))
       
        if not saved_model is None:
            self.gen=self.generate(np.asarray(cond[0]))
            self._restore_scaler(scaler_path1, scaler_path2)
            with tf.Session() as sess:
                condition=tf.placeholder(dtype=tf.float32, shape=[None, 1],
                              name="condition" )
                #norm=tf.placeholder(dtype=tf.bool, shape=(None,1), name="norm")
                #saved=tf.placeholder(dtype=tf.bool, shape=(None,1), name="saved")
                sess.run(tf.global_variables_initializer())
                for i in range(path):
                    print("path ", i)
                    con=cond[0]
                   
                    for j in range(weeks):
                        print("week", j)
                        gen=self.generate(np.asarray(con), n_samples,saved_model=True, inpt_dim=inpt_dim, normalised=True)
                        gen=sess.run(self.gen, feed_dict={condition: np.asarray(con).reshape(-1,1)})
                        sim_cvae[j,:,i]=gen
                        
                        # self.scaler.inverse_transform(gen)
                        # gen=self.generate(np.asarray([con]), saved_model=saved_model)
                        if j !=weeks-1:
                            con=cond[j+1]
                    
        else:
            for i in range(path):
                con=cond[0]
                for j in range(weeks):
                    gen=self.generate(np.asarray([con]), saved_model=None)
                    sim_cvae[j,:,i]= gen[:5]
                    if j !=weeks-1:
                        con=cond[j+1]
        sim_cvae[:,:,i]=self.scaler.inverse_transform(sim_cvae[:,:,i])    
        return sim_cvae
    
    def sim2(self, cond, weeks, path):
        sim_cvae=np.zeros((weeks,5, path))
        for i in range(path):
            con=cond[0]
            for j in range(weeks):
                gen=self.generate(np.asarray([con]), normalised=True, saved_model=None)
                sim_cvae[j,:,i]= gen[:5]
                if j !=weeks-1:
                    con=cond[j+1]
            sim_cvae[:,:,i]=self.scaler.inverse_transform(sim_cvae[:,:,i])
        return sim_cvae
        
    # def sim(self, cond, weeks, path):
    #     sim_cvae=np.zeros((weeks,5, path))
    #     for i in range(path):
            
    #         for j in range(weeks):
    #             gen=self.generate(np.asarray([cond]), normalised=True)
    #             sim_cvae[j,:,i]= gen[:5]
    #             #print(gen[5:])
    #             gen_sc=self.scaler2.inverse_transform(gen[5:].reshape(1,-1))
    #             #print(gen_sc)
    #             gen_sc=np.where(gen_sc<=10.0,10.0,gen_sc)
    #             gen_sc=np.where(gen_sc>=40.0, 40.0, gen_sc)
    #             gen_final=self.scaler2.transform(np.around(gen_sc.astype(np.float64)))
    #             #print(gen_final)
    #             cond=gen_final[0][-1]
    #         sim_cvae[:,:,i]=self.scaler.inverse_transform(sim_cvae[:,:,i])
    #     return sim_cvae
        
        
        