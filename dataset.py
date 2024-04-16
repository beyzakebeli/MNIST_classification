import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MNISTDataset(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Subtract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = os.listdir(data_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Changes the range of the values to [0, 1]
            transforms.Normalize((0.1307,), (0.3081,))  # Subtract mean of 0.1307, and divide by std 0.3081
        ])       

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)

        image = Image.open(img_path)
        image = self.transform(image)
        
        label = int(img_name.split('_')[1].split('.')[0])  # Extracts label
        
        return image, label

# Verifying the implementations
if __name__ == '__main__':
    data_dir = '/home/beyza/other_codes/MNIST_data/extracted_train_data/train'
    dataset = MNISTDataset(data_dir)

    print("Dataset Length:", len(dataset))
    # Takes a sample image and returns the shape and label of the image
    # Which shows the accuracy of the implementations
    img, label = dataset[0]
    print("Sample Image Shape:", img.shape)
    print("Sample Label:", label)

    #### Output 
    # Dataset Length: 60000
    # Sample Image Shape: torch.Size([1, 28, 28])
    # Sample Label: 1
