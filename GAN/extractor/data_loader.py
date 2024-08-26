import torch

from torchvision import datasets
import torchvision.transforms as transforms


def dataset_loader(file_path, batch_size, mnist_size, norm_mean, norm_std):
    transforms_train = transforms.Compose([
        transforms.Resize(mnist_size),
        transforms.ToTensor(),
        transforms.Normalize([norm_mean], [norm_std])
    ])

    train_dataset = datasets.MNIST(file_path, train=True, transform=transforms_train, download=True)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader
