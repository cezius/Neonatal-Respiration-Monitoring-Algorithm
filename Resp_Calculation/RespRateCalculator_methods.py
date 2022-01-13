import numpy as np

def average_vector(flow):
    return [sum(sum(flow))/len(flow)][0]

def calc_rep(flow):
    return np.average(average_vector(flow))

def calcLenght(flow):
  return sum(sum(flow[:,:,0]*flow[:,:,0]))+sum(sum(flow[:,:,1]*flow[:,:,1]))
