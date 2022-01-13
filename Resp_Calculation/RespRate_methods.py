'''
Created on 2022. jan. 2.

@author: nagyn
'''

ws = 200
FPS = 20

import numpy as np
from scipy.fft import fft, fftfreq


from math import sqrt
def RMSE(sig1,sig2):
    if (len(sig1)==len(sig2)):
        mse = []
        for i in range(len(sig1)):
            err = (sig1[i] - sig2[i]) ** 2
            mse.append(err)
        return(sqrt(np.mean(np.asarray(mse))))
    else:
        print("sig1 es sig2 nem egyforma hosszu!")

def makeTresholdCurve(sig,division):
    tresholds = []
    for i in range(0,len(sig),division):
        window = sig[i:(i+division)]
        act_avg = np.average(window)
        for j in range(division):
            tresholds.append(act_avg)
    return tresholds

def tresholdByCurve(sig,tresholds,division,avg):
    sig[sig<avg]=avg
    newsig = []
    for i in range(0,len(sig),20):
        window = sig[i:(i+20)]
        trh = tresholds[i+10]
        window[window<=trh] = avg
        newsig.append(np.asarray(window))
    newsig = np.asarray(newsig)
    newsig = newsig.flatten()
    sig = newsig
    return sig

def calcTreshold2(window,avg):
    upper = []
    for i in range(1,len(window)):
        if window[i-1]<=avg:
            if avg<window[i]:
                upper.append(i)
    return upper

def calcRate(sig):
    hor_treshold = 12  # ami 0.6 sec ami 1.6 HZ ami 100 RPM. Ennel kozelebb nem lehet mert 100-nal magasabb RPM-nel lenne
    
    original = sig[0,:]
    
    tresholds = makeTresholdCurve(sig,20)
    
    # Calculate avg:
    avg = np.average(sig)
    
    # treshold by the treshold curve:
    ujsig = tresholdByCurve(sig,tresholds,20,avg)
    
    # Binarization
    sig[avg<sig] = 1  
    sig[sig<avg] = avg
    
    # relative treshold set by avg:
    upper = calcTreshold2(sig,avg)

    # horizontal treshold
    final_upper = []
    for i in range(0,len(upper)):
      if(abs(upper[i]-upper[i-1])>hor_treshold):
        final_upper.append(upper[i])

    return final_upper,(len(final_upper)/(ws/FPS))*60

def findLocalMaximas(sig,horizontal_treshold):
  ver_treshold = 0.2 # A mynorm-al normalizalt bemeneti jel 0 es 1 kozott van. A peak legyen nagyobb mint 0.3, hogy ertelmes peak-nek lehessen tekinteni ne csak valami ripple-nek
  places = []
  places.append(0)
  if (sig[0]<0):
    jelzo=0               # Ha jelzo=0 akkor negativ, ha jelzo=1 akkor pozitiv es meg nem volt max, ha jelzo=2 akkor pozitiv es volt max
  else:
    jelzo=1
  for i in range(1,len(sig)-1):
    if (sig[i]<0):
      jelzo=0
    if (0<sig[i] and jelzo==0):
      jelzo=1
    if(sig[i-1]<sig[i] and sig[i+1]<sig[i]):
      if (horizontal_treshold<abs(i-places[-1]) and 0<sig[i]):
        if (jelzo==1):
          places.append(i)
          jelzo=2
        else:
          if (sig[places[-1]]<sig[i]):
            places[-1]=i
      else:
        if (sig[places[-1]]<sig[i]):
          places[-1]=i
  if (places[0]==0):
    places = places[1:]
  # vertical treshold
  final_final_upper = []
  for i in range(0,len(places)):
    if(sig[places[i]]>ver_treshold):
      final_final_upper.append(places[i])
  return np.asarray(final_final_upper)

def diffRate(maxes):
  diffs = []
  for i in range(1,len(maxes)):
    diffs.append(maxes[i]-maxes[i-1])
  if (1<len(maxes)):
    return (1.0/(np.mean(diffs)/20.0))*60.0
  else:
    return 0.0

'''
Calc FFT
'''
def calcFFT(sig):
    N = len(sig)
    # sample spacing
    T = 1.0 / FPS
    yf = fft(sig)
    xf = fftfreq(N, T)[:N//2]
    spectrum = 2.0/N * np.abs(yf[0:N//2])
    spectrum = spectrum[(1/3)<xf]
    xf = xf[(1/3)<xf]
    spectrum = spectrum[xf<2.0]
    xx = xf[xf<2.0]
    #plt.plot(xx, spectrum)
    place = np.argmax(spectrum)
    return xx[place]

def mynorm(sig):
  avg = np.mean(sig)
  sig = sig - np.ones_like(sig)*avg
  sig = sig / np.max(sig)
  return sig

def calcRates_with_findLocalMaximas(optsignal,reference):
    diffRate_rates = []
    refs = []
    for i in range(0,len(optsignal)):
      sig = np.asarray(mynorm(optsignal[i]))
      
      maxes = findLocalMaximas(sig,8)
      r =  diffRate(maxes)

      print(i,abs(r-reference[i]))
    
      diffRate_rates.append(r)
      refs.append(reference[i])
    return diffRate_rates,refs

def calcRates_with_calcRate(optsignal,reference):
    diffRate_rates = []
    refs = []
    for i in range(0,len(optsignal)):
      sig = np.asarray(mynorm(optsignal[i]))
      sig = np.reshape(sig, [1,ws])
      
      final_upper,r =  calcRate(sig)
    
      print(i,abs(r-reference[i]))
    
      diffRate_rates.append(r)
      refs.append(reference[i])
    return diffRate_rates,refs

def calcRates_with_FFT(optsignal,reference):
    diffRate_rates = []
    refs = []
    for i in range(0,len(optsignal)):
      sig = np.asarray(mynorm(optsignal[i]))
      
      r = calcFFT(sig)*60
    
      print(i,abs(r-reference[i]))
    
      diffRate_rates.append(r)
      refs.append(reference[i])
    return diffRate_rates,refs





def calcRates_with_findLocalMaximas_and_MA(optsignal,reference):
    len_win = 12
    ma = []
    ma_ref = []
    for i in range(len_win):
      sig = np.asarray(mynorm(optsignal[i]))
    
      maxes = findLocalMaximas(sig,8)
      r =  diffRate(maxes)
      
      ma.append(r)
      ma_ref.append(reference[i])
    
    output_rates = []
    refs = []
    for i in range(len_win+1,len(optsignal)):
      sig = np.asarray(mynorm(optsignal[i]))
    
      maxes = findLocalMaximas(sig,8)
      r =  diffRate(maxes)
      
      del ma[0]
      ma.append(r)
    
      del ma_ref[0]
      ma_ref.append(reference[i])
      
      output_rates.append(np.mean(ma))
      refs.append(np.mean(ma_ref))
    return output_rates,refs

def calcRates_with_calcRate_and_MA(optsignal,reference):
    len_win = 12
    ma = []
    ma_ref = []
    for i in range(len_win):
      sig = np.asarray(mynorm(optsignal[i]))
      sig = np.reshape(sig, [1,ws])
    
      final_upper,r =  calcRate(sig)
      
      ma.append(r)
      ma_ref.append(reference[i])
    
    output_rates = []
    refs = []
    for i in range(len_win+1,len(optsignal)):
      sig = np.asarray(mynorm(optsignal[i]))
      sig = np.reshape(sig, [1,ws])
    
      final_upper,r =  calcRate(sig)
      
      del ma[0]
      ma.append(r)
    
      del ma_ref[0]
      ma_ref.append(reference[i])
      
      output_rates.append(np.mean(ma))
      refs.append(np.mean(ma_ref))
    return output_rates,refs    


def calcRates_with_FFT_and_MA(optsignal,reference):
    len_win = 12
    ma = []
    ma_ref = []
    for i in range(len_win):
      sig = np.asarray(mynorm(optsignal[i]))
      sig = np.reshape(sig, [1,ws])
    
      r =  calcFFT(sig)
      
      ma.append(r)
      ma_ref.append(reference[i])
    
    output_rates = []
    refs = []
    for i in range(len_win+1,len(optsignal)):
      sig = np.asarray(mynorm(optsignal[i]))
      sig = np.reshape(sig, [1,ws])
    
      r =  calcFFT(sig)
      
      del ma[0]
      ma.append(r)
    
      del ma_ref[0]
      ma_ref.append(reference[i])
      
      output_rates.append(np.mean(ma))
      refs.append(np.mean(ma_ref))
    return output_rates,refs


def reshapeSignal(filtered):
    optsignal = []
    for i in range(0,len(filtered),200):
      seq = filtered[i:(i+200)]
      if (len(seq)==200):
        optsignal.append(seq)
    optsignal = np.asarray(optsignal)
    print("Size: " , optsignal.shape)
    return optsignal

def reshapeReference(refflow):
    reference = []
    for i in range(0,len(refflow),200):
      seq = refflow[i:(i+200)]
      if (len(seq)==200):
        reference.append(np.mean(seq))
    reference = np.asarray(reference)
    print("Size: " , reference.shape)
    return reference