'''
Model for 3D INR
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
import pdb
import pywt
from .utils import byte2mb

kernel_cfg = {
	'type': 'SIREN', #'MLP' 
	'no_hidden': 2,
	'no_layers': 1,
	'nonlinearity': 'GELU',   
	# 'norm': 'BatchNorm2d',
	'norm': 'InstanceNorm2d',
	'omega_0': 42,
	'bias': True,
	'size': 3,
	'chang_initialize': True,
	'init_spatial_value': 0.75,
}

conv_cfg = {
	'use_fft': False,
	'bias': True,
	'padding': True,
	'stride': 1,
	'causal': False,
}

ckconv_source =  os.path.join(os.getcwd(), '..')
if ckconv_source not in sys.path:
	sys.path.append(ckconv_source)
sys.path.append("/nobackup/wenxuan/learnable_wavelet/ckconv")
import ckconv


class volumeNet(nn.Module):
	def __init__(self, nlevel, wave, inchannel, outchannel, learnable_wave, transform, mode, use_ckconv=False):
		super(volumeNet, self).__init__()
		self.nlevel = nlevel
		self.wave = wave
		self.mode = mode
		self.learnable_wave = learnable_wave
		self.use_ckconv = use_ckconv

		self.transform = transform                                          
		self.inchannel = inchannel
		self.outchannel = outchannel
		print("model inchannel:", inchannel)
		print("model outchannel:", outchannel)

		if not use_ckconv:
			print("Using vanilla conv (time domain)")
			self.approx_conv1 = nn.Conv3d(inchannel, 64, 3, 1, padding='same')
			self.approx_conv2 = nn.Conv3d(64, 128, 3, 1, padding='same')
			self.approx_conv3 = nn.Conv3d(128, 128, 3, 1, padding='same')
			self.approx_conv4 = nn.Conv3d(128, 64, 3, 1, padding='same')
			self.approx_conv5 = nn.Conv3d(64, outchannel, 3, 1, padding='same')
		else:
			print("Using ckconv (frequency domain)")
			# an additional channel indicating whether the value is known
			inchannel += 1 
			self.approx_conv1 = ckconv.nn.CKConv(inchannel, 64, 2, kernel_cfg, conv_cfg)
			self.approx_conv2 = ckconv.nn.CKConv(64, 128, 2, kernel_cfg, conv_cfg)
			self.approx_conv3 = ckconv.nn.CKConv(128, 128, 2, kernel_cfg, conv_cfg)
			self.approx_conv4 = ckconv.nn.CKConv(128, 64, 2, kernel_cfg, conv_cfg)
			self.approx_conv5 = ckconv.nn.CKConv(64, outchannel, 2, kernel_cfg, conv_cfg)

	
	def forward(self, x):
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