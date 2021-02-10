import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

class CustomDatasetFromImages(Dataset):
    def __init__(self, imageDataset):
        nr_single_examples = 30
        classes = int(imageDataset.NR_OF_IMAGES)
        self.imgdata = np.zeros([nr_single_examples*classes,12,12])
        self.labels = np.zeros([nr_single_examples*classes])

        k = 0
        for i in range(0,nr_single_examples):
            for j in range(0,classes):
                self.imgdata[k,:,:] =  imageDataset.getNoisyImage(j)
                #print('image', k, 'class', j, 'iteration', i)
                #print(self.imgdata[k,:,:])
                self.labels[k] = j
                k = k + 1
        mean = np.mean(self.imgdata)
        std = np.std(self.imgdata)

        self.transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((mean,), (std,))])
        self.label_arr = torch.from_numpy(self.labels).long()

    def __getitem__(self, index):

        self.image_arr_tr = self.transforms(self.imgdata)
        return (self.image_arr_tr[:,index,:].unsqueeze(0), self.label_arr[index])

    def __len__(self):
        return len(self.imgdata)
