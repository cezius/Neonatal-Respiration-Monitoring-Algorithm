'''
Created on 2021. dec. 9.

@author: nagyn
'''

import h5py

# It requires the name f a HDF5. 
# It will unpack the content of the file and return with it
def unpackHDF5(filename):
    hdf5_file = h5py.File(filename,'r')
    hdf5_content = list(hdf5_file.keys())
    print('unpacking the followings:')
    print(hdf5_content)
    frames = hdf5_file['frames']
    labels = hdf5_file['references']
    results = hdf5_file['results']
    return frames, labels, results 
    
# It requires the labels group which is unpacked from HDF5 file and produces the following numpy arrays:
# PPGSignal, PulseNumerical, PulsoxyPPGSignal, PulsoxyPulseNumerical, RespSignal, RespirationNumerical, SpO2Numerical
def getAllLabels(_labels):
    print('our labels are the followings:')
    print(list(_labels.keys()))
    PPGSignal = _labels['PPGSignal']
    PulseNumerical = _labels['PulseNumerical']
    PulsoxyPPGSignal = _labels['PulsoxyPPGSignal']
    PulsoxyPulseNumerical = _labels['PulsoxyPulseNumerical']
    RespSignal = _labels['RespSignal']
    RespirationNumerical = _labels['RespirationNumerical']
    SpO2Numerical = _labels['SpO2Numerical']
    return PPGSignal, PulseNumerical, PulsoxyPPGSignal, PulsoxyPulseNumerical, RespSignal, RespirationNumerical, SpO2Numerical

# It requires the name f a HDF5. 
# It will unpack the content of the file and return with it
def unpackMyHDF5File(filename):
    h5f = h5py.File(filename,'r')
    frames = h5f['Frames'][:]
    labels = h5f['Labels'][:]
    h5f.close()
    return frames,labels
