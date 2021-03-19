import datetime
import numpy as np

from esig import tosig
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utils.leadlag import leadlag
from cvae import CVAE
from rough_bergomi import rough_bergomi

class MarketGenerator:
    def __init__(self, filepath,start=datetime.date(2004, 11, 4),
                 end=datetime.date(2020, 10, 30), freq="M",
                 sig_order=4, rough_bergomi=None):
        self.file=filepath
        
        self.start = start
        self.end = end
        self.freq = freq
        self.order = sig_order

        if rough_bergomi:
             self._load_rough_bergomi(rough_bergomi)
        else:
            self._load_data()

        self._build_dataset()
        self.generator = CVAE(n_latent=8, alpha=0.003)

    def _load_rough_bergomi(self, params):
        grid_points_dict = {"M": 28, "W": 5, "Y": 252}
        grid_points = grid_points_dict[self.freq]
        params["T"] = grid_points / grid_points_dict["Y"]

        paths = rough_bergomi(grid_points, **params)

        self.windows = [leadlag(path) for path in paths]


    def _load_data(self):
       
        data = pd.read_excel(self.file, header=0, sheet_name='SP 500', usecols=['Date','VIX','SP500'] , skiprows=[1])
        data.Date=pd.to_datetime(data.Date)
        data.set_index('Date', inplace=True)
        self.data=data
        #get number of risk factors
        self.rf=len(list(self.data))
        #self.data.index = pd.to_datetime(self.data.index.astype(str))
        self.windows = []
        #resample goes after true calender days to sample the correct week/ month.Missing days are ignored (the window is then simply shorter)
        for _, window in self.data.resample(self.freq):
            #value shape is time window x number of Risk factors to be simulated
            values = window.values# / window.values[0]
            path = leadlag(values)
            self.windows.append(path)
        #shape of path is LAG_FACTOR1, LAG_FACTOR2, LEAD_FACTOR1,LEAD_FACTOR2 etc as list of numpy arrays, each numpy array being of shape 2*frequncy length (as lags are repeated) x 2* number of RF (as each RF gets lead and lag part)
    def _logsig(self, path):
        return tosig.stream2logsig(path, self.order)

    def _build_dataset(self):
        if self.order:
            self.orig_logsig = np.array([self._logsig(path) for path in tqdm(self.windows, desc="Computing log-signatures")])
        else:
            #to get real values without repititions etc take every second element from lag part. lag part has columns of RF1, RF2 etc.
            #ATTENTION : during excel reading the columns get sorted alphabetically!!
            #take all columns of RF for joint simulation 0:numberRF +1 to include all columns
            
            #NOTE: only return does not make use of lead lag transformation, as taking every second element in log part corresponds to taking initial values 
            #each path in windows has 2*frequency x rf*2 shape. taking every second element rowwise gives exactly frequency amount in rows.
            
            #print("shape of one window element: "+ str(self.windows[0].shape))
            
            #need to specify axis=0 in case, cannot reshape now because otherwise the cuts might be wrong
            #gibes np array with each entry being 2D np array
            
            self.orig_logsig = np.array([np.diff(np.log(path[::2,self.rf:]), axis=0) for path in self.windows], dtype=object)
            #orders change, now second column is SPX, first is VIX
            
            print("shape logsig 0 after first trandform: "+ str(self.orig_logsig[0].shape))
            #print(self.orig_logsig[0][0:5,:])
            #fill array with lists of piecewise paths after leadlag and frequency resample procedure
            self.orig_logsig = np.array([p for p in self.orig_logsig if p.shape[0] >= 4], dtype=object)
            
            print("shape logsig 0 after second trandform: "+ str(self.orig_logsig[0].shape))
            
            steps = min(map(len, self.orig_logsig))
            print("shortest snippet: {}".format(steps))
            #cut all paths to the shortest path length
            self.orig_logsig = np.array([val[:steps,:] for val in self.orig_logsig], dtype=object)
            
            print("shape logsig 0 after cuts to shortest frequency snippet: "+ str(self.orig_logsig[0].shape))
            
            #reshape so that each row contains all risk factors  (each rf of length frequency)
            #data is given rowwise to CVAE, let column dimension be 0 to prevent minmaxscalar encountering 3D array
            #need to transpose bec else val is [[SPX1,VIX1]; [SPX2, VIX2]] and reshape would read it into [SPX1, VIX1, SPX2,VIX2] etc
            self.orig_logsig = np.array([val.T.reshape(self.rf*val.shape[0],) for val in self.orig_logsig], dtype=object)
            
            print("shape logsig 0 after reshapeing: "+ str(self.orig_logsig[0].shape)) 
            print("shape entire logsig after reshapeing: "+ str(self.orig_logsig.shape))
        self.scaler = MinMaxScaler(feature_range=(0.00001, 0.99999))
        logsig = self.scaler.fit_transform(self.orig_logsig)

        self.logsigs = logsig[1:]
        self.conditions = logsig[:-1]


    def train(self, n_epochs=10000):
        self.generator.train(self.logsigs, self.conditions, n_epochs=n_epochs)

    def generate(self, logsig, n_samples=None, normalised=False):
        generated = self.generator.generate(logsig, n_samples=n_samples)

        if normalised:
            return generated

        if n_samples is None:
            return self.scaler.inverse_transform(generated.reshape(1, -1))[0]

        return self.scaler.inverse_transform(generated)
