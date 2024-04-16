import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image


# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layer_stack(x)
        return logits


def train_model(dataloader, model, loss_fn, optimizer, epochs=5):
    model.train()  # Set the model to training mode
    size = len(dataloader.dataset)
    for epoch in range(epochs):
        correct = 0
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        accuracy = 100 * correct / size
        print(f"Epoch {epoch + 1}: Accuracy: {accuracy:>0.1f}%")


def test_model(dataloader, model):
    model.eval()  # Set the model to evaluation mode
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += nn.CrossEntropyLoss()(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    size = len(dataloader.dataset)
    accuracy = 100 * correct / size
    test_loss /= size
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")


def classify_image(model, file_path):
    # Load and preprocess the image
    image = Image.open(file_path)
    image = image.convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize like the training data
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        prediction = model(image)
    return torch.argmax(prediction, dim=1).item()


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.MNIST('/home/sanele/Downloads/MNIST data (1)', download=False, train=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = datasets.MNIST('/home/sanele/Downloads/MNIST data (1)', download=False, train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
    return train_loader, test_loader


def main():
    train_loader, test_loader = load_data()
    model = NeuralNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.1)
    print("Training model...")
    train_model(train_loader, model, nn.CrossEntropyLoss(), optimizer)
    print("Testing model...")
    test_model(test_loader, model)

    print("Done!")
    while True:
        filepath = input("Please enter a filepath ('exit' to quit): ")
        if filepath.lower() == 'exit':
            print("Exiting...")
            break
        prediction = classify_image(model, filepath)
        print(f'Classifier: {prediction}')


if __name__ == '__main__':
    main()
