import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Carregar o conjunto de dados MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 2. Definir a rede neural
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Entrada 28x28 pixels (784), saída 128
        self.fc2 = nn.Linear(128, 64)     # Camada oculta
        self.fc3 = nn.Linear(64, 10)      # Saída: 10 classes (0-9)

    def forward(self, x):
        x = x.view(-1, 28*28)  # "Achatar" a imagem 28x28 em um vetor 1D de 784 elementos
        x = torch.relu(self.fc1(x))  # Função de ativação ReLU
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Saída sem ativação (softmax será feito na função de perda)
        return x

model = SimpleNN()

# 3. Definir a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()  # Para problemas de classificação
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Treinar o modelo
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Colocar o modelo no modo de treinamento
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zerando o gradiente das iterações anteriores
        outputs = model(images)  # Passando as imagens pela rede
        loss = criterion(outputs, labels)  # Calculando a perda
        loss.backward()  # Backpropagation (ajustando os pesos)
        optimizer.step()  # Atualizando os pesos

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Pegando a previsão (classe com maior probabilidade)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct/total}%")

# 5. Avaliar a IA com os dados de teste
model.eval()  # Colocar o modelo no modo de avaliação
correct = 0
total = 0

with torch.no_grad():  # Desabilitar o cálculo do gradiente para eficiência
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total}%")

# Visualizar algumas previsões
dataiter = iter(test_loader)  # Criando o iterador para o DataLoader
images, labels = next(dataiter)  # Obtendo o próximo lote de imagens e rótulos
outputs = model(images)  # Passando as imagens pelo modelo

_, predicted = torch.max(outputs, 1)  # Obtendo a classe com maior probabilidade

# Exibir as primeiras 5 imagens e suas previsões
fig, axes = plt.subplots(1, 5, figsize=(12, 4))  # Criar um grid de 1x5 para exibir as imagens
for i in range(5):
    axes[i].imshow(images[i].numpy().squeeze(), cmap='gray')  # Exibir a imagem (remove a dimensão de canal)
    axes[i].set_title(f'Pred: {predicted[i].item()}')  # Exibir o rótulo previsto
    axes[i].axis('off')  # Desativa os eixos para melhor visualização

plt.show()
