import numpy as np 
import torch
from utils import *
from dataloaders import *
from model import *
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


def train_epoch(vae,device,loader,optim):

	vae.train()
	train_loss = 0.0

	#Iterate the dataloader (we do not need the label values, this is unsupervised learning)
	#print(train_loader.shape)
	for x, y in loader:
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
		#print('\t partial train loss (single batch): %f' % (loss.item()))
		train_loss+=loss.item()

	return train_loss / len(train_loader)

def test_epoch(vae,device,test_set):

	vae.eval()

	res = []

	with torch.no_grad():

		for x in test_set:
			x = x.float().to(device)
			x_hat = vae(x)
			res.append(x_hat)
	return res



if __name__ == "__main__":
	test_flag = True
	train_flag = False

	device = get_device().device

	vae = VariationalAutoencoder(latent_dims=20).to(device)

	lr = 1e-3

	optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

	num_epochs = 300

	

	if train_flag:
		print(train)
		train, test = DataSet("data").load(train_size = 500, test_size = 500)
		train_loader = dataloader(batch_size = 10).create_loader(train)
		for epoch in tqdm.tqdm(range(num_epochs)):
		   train_loss = train_epoch(vae,device,train_loader,optim)
		   print('\n EPOCH {}/{} \t train loss {:.3f}\t '.format(epoch + 1, num_epochs,train_loss))
		   #plot_ae_outputs(vae.encoder,vae.decoder,n=10)

		torch.save(vae,"vae.pt")

	if test_flag:
		train, test = DataSet("data").load(train_size = 1000, test_size = 1000)
		train_loader = dataloader(batch_size = 1).create_loader(train, shuffle= False)

		vae = torch.load("vae.pt")

		vae.eval()
		count = 0

		for x,y in train_loader:

			x = x.float().to(device)

			x_hat = vae(x)

			y.float().to(device)
			print(y.shape)

			transform = transforms.ToPILImage()

			y = transform(y.squeeze())
			x_hat = transform(x_hat.squeeze())
			x = transform(x.squeeze())
			count+=1
			x.save(f"Predicted_Masks/{count}OG.png")
			y.save(f"Predicted_Masks/{count}OG_mask.png")
			x_hat.save(f"Predicted_Masks/{count}Pred.png")
			





