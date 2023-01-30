import numpy as np 
import torch
import torchvision 
from torch.utils.data import DataLoader
from utils import *

class dataloader(object):

	def __init__(self, batch_size = 1):

		self.batch_size = batch_size

	def create_loader(self,dataset,shuffle = True):
		loader = DataLoader(dataset,batch_size = self.batch_size, shuffle = shuffle)
		return loader

if __name__ == "__main__":
	from argparse import ArgumentParser
	parser = ArgumentParser(add_help=True)
	def test_load():
		train, test = DataSet("data").load(train_size = 1, test_size = 1)
		train_loader = []
		try:
			train_loader = dataloader(batch_size = 10).create_loader((test))
			assert(train_loader != [])
			print(train_loader)
			print("DataLoader created!!")
		except:
			print("Failed to create DataLoader. :(")



	parser.add_argument("--test_load", action='store_true',
						help="To test the creation of the dataloader")

	args = parser.parse_args()

	if args.test_load:
		test_load()
