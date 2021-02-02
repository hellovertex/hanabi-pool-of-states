# iterable dataset -> dataloader (yields items from dataset iterator)
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

dataset = torchvision.datasets.MNIST(root='./garbage', download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=2)
# wont work: print(next(loader))
loader = iter(loader)
print(type(loader))
print(hasattr(loader, '__iter__'))
print(hasattr(loader, '__next__'))

states = np.tile(np.array([[0, 1, 2, 3, 3, 4], [5, 6, 6, 7, 8, 9]]), (2, 1))


class MapStyleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(MapStyleDataset).__init__()
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MapStyleLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batchsize=1):
        super(MapStyleLoader).__init__()
        self.dataset = dataset
        self.batchsize = batchsize


mapstyle_dataset = MapStyleDataset(states)
# loader = MapStyleLoader(dataset)  #  --> AttributeError 'persistent workers'
mapstyle_loader = torch.utils.data.DataLoader(mapstyle_dataset, batch_size=2)
print(hasattr(mapstyle_loader, '__iter__'), hasattr(mapstyle_loader, '__next__'))
for val in mapstyle_loader:
    print(val)


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data):
        super(IterableDataset).__init__()
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data

    def __iter__(self):
        return iter(self.data)


iterable_dataset = IterableDataset(states)
# iterable_loader = MapStyleLoader(dataset)  #  --> AttributeError 'persistent workers'
iterable_loader = iter(torch.utils.data.DataLoader(iterable_dataset, batch_size=2))
print(hasattr(iterable_loader, '__iter__'), hasattr(iterable_loader, '__next__'))
for val in iterable_loader:
    print(val)


# conclusion is that Dataloader does not return a generator but just an iterator

class IterableStatesCollection(torch.utils.data.IterableDataset):
    def __init__(self, states=None):
        super(IterableStatesCollection).__init__()
        if not states:
            states = []
        self.states = states

    def __iter__(self):
        # collect some states in a list (streaming data)
        # doing this using __len__ and __getitem__ would also work:
        # __getitem__ should implement a check if num_states have been collected
        # and load new states otherwise but its not as nice as an iterable
        states = []
        return iter(states)

# the iter() call of the dataset streams data to the loader who can batch, shuffle, transform it
# to create a new iterator, and by calling iter() on the loader it returns an instance of
# torch.utils.data._BaseDataLoaderIter which happens to implement __next__ and is thus a generator

#train_loader = torch.utils.data.DataLoader(dataset=IterableStatesCollection, batch_size=1)
#test_loader = torch.utils.data.DataLoader(dataset=IterableStatesCollection, batch_size=1)
#gen_train, gen_test = iter(train_loader), iter(test_loader)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size, images.shape)
