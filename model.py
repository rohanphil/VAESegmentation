import torch
import numpy as np 
from torch import nn
import torch.nn.functional as F 
import torch.optim as optim
from utils import *
from dataloaders import *

class Encoder(nn.Module):
	def __init__(self, latent_dim):

		super(Encoder, self).__init__()
		self.c1 = nn.Conv2d(3,8,3, stride = 2, padding = 1)
		self.c2 = nn.Conv2d(8,16,3, stride = 2, padding = 1)
		self.batch2 = nn.BatchNorm2d(16)
		self.c3 = nn.Conv2d(16,32,3, stride = 2, padding = 0)
		self.l1 = nn.Linear(4608,128)
		self.l2 = nn.Linear(128,latent_dim)
		self.l3 = nn.Linear(128, latent_dim)

		self.N = torch.distributions.Normal(0,1)
		self.N.loc = self.N.loc.cuda()
		self.N.scale = self.N.scale.cuda()
		self.kl = 0



	def forward(self,x):

		x = F.relu(self.c1(x))
		#print(f"c1 enc shape:{x.shape}")
		x = F.relu(self.batch2(self.c2(x)))
		#print(f"c2 enc shape:{x.shape}")
		x = F.relu(self.c3(x))
		#print(f"c3 enc shape:{x.shape}")
		#print(x.shape)
		x = torch.flatten(x, start_dim=1)
		x = F.relu(self.l1(x))
		mu = self.l2(x)
		sigma = torch.exp(self.l3(x))
		z = mu + sigma*self.N.sample(mu.shape)
		self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
		return z


class Decoder(nn.Module):
	def __init__(self, latent_dim):
		super().__init__()

		self.linear_dec = nn.Sequential(
			nn.Linear(latent_dim,128),
			nn.ReLU(),
			nn.Linear(128,4608),
			nn.ReLU()
			)

		self.unflatten = nn.Unflatten(dim = 1, unflattened_size = (32,12,12))

		self.decoder_conv = nn.Sequential(
			nn.ConvTranspose2d(32,16,3,stride = 2, output_padding = 0),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.ConvTranspose2d(16,8,3,stride = 2, padding = 1, output_padding = 1),
			nn.BatchNorm2d(8),
			nn.ReLU(),
			nn.ConvTranspose2d(8,1,3,stride = 2, padding = 0, output_padding = 0)
			)

		# self.ct1 = nn.ConvTranspose2d(32,16,3,stride = 2, output_padding = 0)
		# self.bt1 = nn.BatchNorm2d(16)
		# self.ct2 = nn.ConvTranspose2d(16,8,3,stride = 2, padding = 1, output_padding = 1)



	def forward(self, x):
		x = self.linear_dec(x)
		x = self.unflatten(x)
		x = self.decoder_conv(x)
		x = torch.sigmoid(x)
		return x


class VariationalAutoencoder(nn.Module):
	def __init__(self, latent_dims):
		super(VariationalAutoencoder, self).__init__()
		self.encoder = Encoder(latent_dims)
		self.decoder = Decoder(latent_dims)

	def forward(self, x):
		z = self.encoder(x)
		return self.decoder(z)



if __name__ == "__main__":


	train, test = DataSet("data").load(train_size = 2, test_size = 2)
	# train = torch.stack(train, dim = 0)
	# mask = torch.stack(mask, dim = 0)
	# print(f"train: {train.shape}, mask : {mask.shape}")
	train_loader = dataloader(batch_size = 10).create_loader(train)
	device = get_device().device

	lr = 1e-3 

	d = 4

	vae = VariationalAutoencoder(latent_dims=d).to(device)

	optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

	vae.train()
	train_loss = 0.0

	#Iterate the dataloader (we do not need the label values, this is unsupervised learning)
	#print(train_loader.shape)
	for x, y in train_loader:
		#print(x.shape, y.shape) 
		# Move tensor to the proper device
		x = x.float().to(device)
		y = y.float().to(device)
		x_hat = vae(x)
		#print(x_hat.shape)
		# Evaluate loss
		#print(y)
		loss = ((y - x_hat)**2).sum() + vae.encoder.kl

		# Backward pass
		optim.zero_grad()
		loss.backward()
		optim.step()
		# Print batch loss
		print('\t partial train loss (single batch): %f' % (loss.item()))
		train_loss+=loss.item()

	print(train_loss / len(train_loader))









#To dos

# 1) Flesh out encoder and decoder
# 2) Do math to calculate reconstruction loss 
# 3) Write tests 
# 4) Deploy to github


