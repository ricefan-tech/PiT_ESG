import numpy as np

def leadlag(X):
    lag = []
    lead = []

    for val_lag, val_lead in zip(X[:-1], X[1:]):
        lag.append(val_lag)
        lead.append(val_lag)

        lag.append(val_lag)
        lead.append(val_lead)

    lag.append(X[-1])
    lead.append(X[-1])
    
    #output of numpy c_ of list of tuples will be:
    # lag1 (x,y), lag1 (x,y)
    # lag1 (x,y), lead1(x,y)
    # lag2(x,y) =lead1, lag2
    #lag2, öead 2 etc
    #hence for full graphs take every second element from the lag part 
    return np.c_[lag, lead]