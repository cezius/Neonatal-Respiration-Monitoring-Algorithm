import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#!/usr/bin/env python
# coding: utf-8
#get_ipython().run_line_magic('pylab', 'inline')
 
#!/usr/bin/env python
# coding: utf-8
#get_ipython().run_line_magic('pylab', 'inline')
 
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/17cbfe0b68148d129a3ddaa227696496
    @author: wassname
    """
    intersection= (y_true * y_pred).abs().sum(dim=(2,3))
    sum_ = torch.sum(y_true.abs() + y_pred.abs(), dim=(2,3))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return torch.mean((1 - jac) * smooth)


# We are using DoubleConv In many places in the architechture. So first we will define this one.
# This will be the class double conv
class DoubleConv(nn.Module): # If we want to use a class as a building block of an architechture the we have to inherit nn.Module
    def __init__(self, in_channels, out_channels):
        super().__init__() # First we have to initialize the ancestor
        self.double_conv = nn.Sequential( # nOW WE WILL define the double_conv which will contain the 2 convolutions with batchnorm
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
 
 
# First lets define a class for creating downsampling layers: 
class Down(nn.Module): 
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels): # This will wait for two inputs: the number of in_channels and ...
        super().__init__() 
        self.maxpool_conv = nn.Sequential( # After we will define maxpool_conv which consist of a..
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )
    def forward(self, x): # In the forward prop we will apply this on the input x.
        return self.maxpool_conv(x)
 
# After lets define a class for UP layers
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels): 
        super().__init__() # First we have to initialize the ancestor
 
        # After lets define the parts of up layer. In this we have
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=3, stride=2)
            #the floor division // rounds the result down to the nearest whole number
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2): # In the forward prop we have two inputs: one is coming from down (x1) and one
                               # is coming from the skip connection (x2)
        x1 = self.up(x1) # We will apply the transpose conv o the x1
 
        # NOw we will calculate what is the difference in x and y direction btween the size of x1 and x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # It is important to calculate how mutch padding we needin the different 
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
 
        # wITH TORCH cat we can concatenate the two tensor
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
 
# Finally we will define a class for the outCon which
# will take us back to the out_channel num.
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)



class UNET(nn.Module):
  def __init__(self, i_ch, o_ch):
    super().__init__()
 
    # downsampling
    self.inc = DoubleConv(i_ch, 64) #i_ch = input_channel
    self.down1 = Down(64,128)
    self.down2 = Down(128,256)
    self.down3 = Down(256,512)
    self.down4 = Down(512,1024)
 
    # upsampling
    self.up1 = Up(1024, 512)
    self.up2 = Up(512, 256)
    self.up3 = Up(256, 128)
    self.up4 = Up(128, 64)
    self.outconv = OutConv(64, o_ch) #o_ch = out_channel
 
  def forward(self, x):
    x1 = self.inc(x) 
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
 
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    x = self.outconv(x)
 
    return x

def processOuput(out_mask,treshold):
  array = out_mask.cpu().detach().numpy()
  array = np.swapaxes(array,1,3)

  array = array[0,:,:,:]
  array = cv2.cvtColor(array,cv2.COLOR_RGB2GRAY)

  m = array
  m[m<treshold] = 0
  m[treshold<m] = 1
  return m

def callAutomaticUNET(img,wh,net,dev):
  img = np.cast['float32']((img/np.max(img)))
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  img = cv2.resize(img, (wh,wh))
  input_images=[]
  input_images.append(img)
  imgs = np.asarray(input_images)
  imgs = np.swapaxes(imgs,1,3)
  im_tensor = torch.from_numpy(imgs)

  img_batch = im_tensor.to(dev,dtype=torch.float32)
  mask_pred = net(img_batch)
  processed_mask = processOuput(mask_pred,200)
  return processed_mask
