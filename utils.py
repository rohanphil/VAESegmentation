import torch
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm
from torch.utils.data import DataLoader, random_split, Dataset


class TrainSet(Dataset):

	def __init__(self, inputs, labels):
		self.inputs = inputs
		self.labels = labels

	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, idx):
		return self.inputs[idx], self.labels[idx]


class DataSet(object):

	def __init__(self, datapath):

		self.train = None
		self.test = None
		self.masks = None
		self.path  = datapath
		self.train_transform = transforms.Compose([
			lambda x: x.convert("RGB"),
			transforms.PILToTensor(),
			lambda x: x/255.
			])
		self.mask_transform = transforms.Compose([
			lambda x: x.convert("L"),
			transforms.PILToTensor(),
			lambda x: x/255.
			])

	def load(self, train_size = 500, test_size = 100):
		try:
			assert(train_size<=4001)
			assert(test_size<=18001)
		except:
			print("Specified train or test size is invalid. Make sure they are within the ranges of (1-4000) and (1-18000) respectively")
		train_image_list = glob.glob(self.path+'/train/images/'+'*.png')[:train_size]
		train_mask_list = glob.glob(self.path+'/train/masks/' + '*.png')[:train_size]
		test_image_list = glob.glob(self.path+'/test/images/'+'*.png')[:test_size]
		#print(train_image_list)
		self.train = [self.train_transform(Image.open(f, 'r')) for f in tqdm.tqdm(train_image_list)]
		self.masks = [self.mask_transform(Image.open(f, 'r')) for f in tqdm.tqdm(train_mask_list)]
		self.test = [self.train_transform(Image.open(f,'r')) for f in tqdm.tqdm(test_image_list)]
		#print(type(self.train[0]))

		return TrainSet(self.train,self.masks), self.test

class get_device(object):

	def __init__(self):
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
	trainSet, test = DataSet("data").load(train_size = 1, test_size = 1)
	print(type(trainSet))

	

