import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MNISTDataset
from model import LeNet5, CustomMLP
import matplotlib.pyplot as plt


def train(model, trn_loader, device, criterion, optimizer, weight_decay=0.0001):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim
        weight_decay: L2 regularization strength (default: 0.001)

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    model.train()
    trn_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trn_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Calculates L2 regularization term
        l2_regularization = 0.0
        for param in model.parameters():
            l2_regularization += torch.norm(param, p=2)**2  
            
        loss += 0.5 * weight_decay * l2_regularization  # Adds regularization term to the loss

        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    trn_loss /= len(trn_loader)
    
    return trn_loss, acc
    

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    model.eval()  
    tst_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tst_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            tst_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    tst_loss /= len(tst_loader)

    return tst_loss, acc

def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_dir = '/home/beyza/other_codes/MNIST_data/extracted_train_data/train'
    test_data_dir = '/home/beyza/other_codes/MNIST_data/extracted_test_data/test'
    train_dataset = MNISTDataset(train_data_dir)
    test_dataset = MNISTDataset(test_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    lenet_model = LeNet5().to(device)
    mlp_model = CustomMLP().to(device)

    optimizer = optim.SGD(lenet_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 5

    #This part of the code from here is for obtaining the performance graphs

    lenet_train_losses = []
    lenet_train_accs = []
    lenet_test_losses = []
    lenet_test_accs = []

    mlp_train_losses = []
    mlp_train_accs = []
    mlp_test_losses = []
    mlp_test_accs = []

    for epoch in range(num_epochs):
        trn_loss, trn_acc = train(lenet_model, train_loader, device, criterion, optimizer)
        tst_loss, tst_acc = test(lenet_model, test_loader, device, criterion)
        
        lenet_train_losses.append(trn_loss)
        lenet_train_accs.append(trn_acc)
        lenet_test_losses.append(tst_loss)
        lenet_test_accs.append(tst_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  LeNet-5: Training Loss={trn_loss:.4f}, Accuracy={trn_acc:.2f}%')
        print(f'  LeNet-5: Test Loss={tst_loss:.4f}, Accuracy={tst_acc:.2f}%')

    optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(num_epochs):
        trn_loss, trn_acc = train(mlp_model, train_loader, device, criterion, optimizer)
        tst_loss, tst_acc = test(mlp_model, test_loader, device, criterion)

        mlp_train_losses.append(trn_loss)
        mlp_train_accs.append(trn_acc)
        mlp_test_losses.append(tst_loss)
        mlp_test_accs.append(tst_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  CustomMLP: Training Loss={trn_loss:.4f}, Accuracy={trn_acc:.2f}%')
        print(f'  CustomMLP: Test Loss={tst_loss:.4f}, Accuracy={tst_acc:.2f}%')

    epochs = range(1, num_epochs + 1, 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, lenet_train_losses, label='LeNet-5 Training')
    plt.plot(epochs, lenet_test_losses, label='LeNet-5 Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LeNet-5 Training and Test Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, lenet_train_accs, label='LeNet-5 Training')
    plt.plot(epochs, lenet_test_accs, label='LeNet-5 Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('LeNet-5 Training and Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mlp_train_losses, label='CustomMLP Training')
    plt.plot(epochs, mlp_test_losses, label='CustomMLP Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CustomMLP Training and Test Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, mlp_train_accs, label='CustomMLP Training')
    plt.plot(epochs, mlp_test_accs, label='CustomMLP Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('CustomMLP Training and Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()