import torchvision.datasets as datasets
import torchvision
import torch

# simple data loader for pytorch 
def load_data():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # convert to tensor 
        ])
        
    train = datasets.MNIST(root='./data', train = True, download = True, transform = transform)
    test = datasets.MNIST(root='./data', train = False, download = True, transform = transform)

    return train, test

if __name__ == "__main__":
    train, test = load_data()
    img, label = train[0]
    print(torch.max(img))