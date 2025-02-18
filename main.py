import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

# 1. Carregar o conjunto de dados MNIST
transformacao = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

dados_treinamento = datasets.MNIST(root='./data', train=True, download=True, transform=transformacao)
dados_teste = datasets.MNIST(root='./data', train=False, download=True, transform=transformacao)

carregador_treinamento = DataLoader(dados_treinamento, batch_size=64, shuffle=True)
carregador_teste = DataLoader(dados_teste, batch_size=64, shuffle=False)

# 2. Definir a rede neural
class RedeNeuralSimples(nn.Module):
    def __init__(self):
        super(RedeNeuralSimples, self).__init__()
        self.camada_oculta1 = nn.Linear(28*28, 128)  # Entrada 28x28 pixels (784), saída 128
        self.camada_oculta2 = nn.Linear(128, 64)     # Segunda camada oculta
        self.camada_saida = nn.Linear(64, 10)       # Saída: 10 classes (dígitos de 0 a 9)

    def forward(self, entrada):
        entrada = entrada.view(-1, 28*28)  # "Achatar" a imagem 28x28 em um vetor 1D de 784 elementos
        entrada = torch.relu(self.camada_oculta1(entrada))  # Função de ativação ReLU
        entrada = torch.relu(self.camada_oculta2(entrada))
        saida = self.camada_saida(entrada)  # Saída sem ativação (softmax será feito na função de perda)
        return saida

modelo = RedeNeuralSimples()

# 3. Definir a função de perda e o otimizador
funcao_perda = nn.CrossEntropyLoss()  # Para problemas de classificação
otimizador = optim.SGD(modelo.parameters(), lr=0.01)

# 4. Treinar o modelo
num_epocas = 5
for epoca in range(num_epocas):
    modelo.train()  # Colocar o modelo no modo de treinamento
    perda_acumulada = 0.0
    acertos = 0
    total_amostras = 0
    
    for imagens, rotulos in carregador_treinamento:
        otimizador.zero_grad()  # Zerando o gradiente das iterações anteriores
        saidas = modelo(imagens)  # Passando as imagens pela rede
        perda = funcao_perda(saidas, rotulos)  # Calculando a perda
        perda.backward()  # Backpropagation (ajustando os pesos)
        otimizador.step()  # Atualizando os pesos

        perda_acumulada += perda.item()
        _, previsao = torch.max(saidas, 1)  # Pegando a previsão (classe com maior probabilidade)
        total_amostras += rotulos.size(0)
        acertos += (previsao == rotulos).sum().item()

    print(f"Época {epoca+1}/{num_epocas}, Perda: {perda_acumulada/len(carregador_treinamento)}, Acurácia: {100 * acertos/total_amostras}%")

# 5. Avaliar a IA com os dados de teste
modelo.eval()  # Colocar o modelo no modo de avaliação
acertos = 0
total_amostras = 0

with torch.no_grad():  # Desabilitar o cálculo do gradiente para eficiência
    for imagens, rotulos in carregador_teste:
        saidas = modelo(imagens)
        _, previsao = torch.max(saidas, 1)
        total_amostras += rotulos.size(0)
        acertos += (previsao == rotulos).sum().item()

print(f"Acurácia no teste: {100 * acertos / total_amostras}%")

# Visualizar algumas previsões
lote_teste = random.choice(list(carregador_teste))  # Escolhe um batch aleatório
imagens, rotulos = lote_teste
saidas = modelo(imagens)  # Passando as imagens pelo modelo

_, previsao = torch.max(saidas, 1)  # Obtendo a classe com maior probabilidade

# Exibir as primeiras 5 imagens e suas previsões
figura, eixos = plt.subplots(1, 5, figsize=(12, 4))  # Criar um grid de 1x5 para exibir as imagens
for i in range(5):
    eixos[i].imshow(imagens[i].numpy().squeeze(), cmap='gray')  # Exibir a imagem (remove a dimensão de canal)
    eixos[i].set_title(f'Previsto: {previsao[i].item()}')  # Exibir o rótulo previsto
    eixos[i].axis('off')  # Desativa os eixos para melhor visualização

plt.show()
