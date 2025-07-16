from torchvision import datasets, transforms

def get_dataset(dir, name):
    if name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = datasets.MNIST(dir, train=True, download=True, transform=transform)
        test = datasets.MNIST(dir, train=False, transform=transform)
    elif name == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
        test = datasets.CIFAR10(dir, train=False, transform=transform_test)
    return train, test
