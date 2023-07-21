'''
Model for 3D INR
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_wavelets import Learnable2D
import math
import os
import sys
import pdb
import pywt
from pytorch_wavelets import DWTForward, DWTInverse

class volumeNet(nn.Module):
	def __init__(self, nlevel, wave, inchannel, outchannel, learnable_wave, transform, mode):
		super(volumeNet, self).__init__()
		self.nlevel = nlevel
		self.wave = wave
		self.mode = mode
		self.learnable_wave = learnable_wave
		
		self.transform = transform                                          
		self.inchannel = inchannel
		self.outchannel = outchannel
		print("model inchannel:", inchannel)
		print("model outchannel:", outchannel)

		self.approx_conv1 = nn.Conv3d(inchannel, 64, 3, 1, padding='same')
		self.approx_conv2 = nn.Conv3d(64, 128, 3, 1, padding='same')
		self.approx_conv3 = nn.Conv3d(128, 128, 3, 1, padding='same')
		self.approx_conv4 = nn.Conv3d(128, 64, 3, 1, padding='same')
		self.approx_conv5 = nn.Conv3d(64, outchannel, 3, 1, padding='same')

	def forward(self, x, verbose=False, autocast=False):
		return self._forward(x)
	
	@torch.autocast("cuda")
	def _forward(self, x):
		la = x
		la = self.approx_conv1(la)
		# la = F.relu(la)
		la = torch.sin(la)
		la = self.approx_conv2(la)
		# la = F.relu(la)
		la = torch.sin(la)
		la = self.approx_conv3(la)
		# la = F.relu(la)
		la = torch.sin(la)
		la = self.approx_conv4(la)
		# la = F.relu(la)
		la = torch.sin(la)
		la = self.approx_conv5(la)
		signal = la
		return signal