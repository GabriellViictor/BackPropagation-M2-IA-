import numpy as np
import matplotlib.pyplot as plt

# Função Sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Normalização dos dados
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Função de Treinamento
def train(X, y, hidden_neurons=10, learning_rate=0.1, epochs=20000):
    input_neurons = X.shape[1]
    output_neurons = 1

    # Inicialização aleatória dos pesos e bias
    np.random.seed(42)  # Para garantir reprodutibilidade
    weights_input_hidden = np.random.randn(input_neurons, hidden_neurons)
    weights_hidden_output = np.random.randn(hidden_neurons, output_neurons)
    bias_hidden = np.zeros((1, hidden_neurons))
    bias_output = np.zeros((1, output_neurons))

    # Loop de treinamento
    for epoch in range(epochs):
        # Forward Pass
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)

        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(final_input)

        # Erro e Backpropagation
        error = y - predicted_output
        d_output = error * sigmoid_derivative(predicted_output)

        error_hidden = d_output.dot(weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_output)

        # Atualização dos pesos e bias
        weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
        weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

        # Exibir perda a cada 1000 épocas
        if epoch % 1000 == 0:
            loss = np.mean(np.abs(error))
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Função de Previsão
def predict(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(final_input)
    return predicted_output

# Geração da Onda Senoidal e preparação dos dados
x = np.linspace(0, 2 * np.pi, 100)
sine_wave = np.sin(x)
sine_wave_normalized = normalize(sine_wave)

# Preparação dos dados para treinamento
window_size = 5
X_train = np.array([sine_wave_normalized[i:i + window_size] for i in range(len(sine_wave_normalized) - window_size)])
y_train = sine_wave_normalized[window_size:].reshape(-1, 1)

# Treinamento do Modelo
weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train(X_train, y_train, epochs=10000)

# Previsão
predicted = predict(X_train, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

# Plotando os resultados
plt.plot(sine_wave_normalized, label='Seno Original (Normalizado)')
plt.plot(np.concatenate([sine_wave_normalized[:window_size], predicted.flatten()]), label='Seno Predito')
plt.legend()
plt.show()
