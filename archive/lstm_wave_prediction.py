import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def generate_data(num_sequences=1000, sequence_length=50):
    # Generate 1000 random sine and cosine sequences
    # We will predict the next value in the sequence
    X = []
    y = []
    
    for _ in range(num_sequences):
        # Random frequency, phase, and amplitude
        freq = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.5, 1.5)
        
        # Randomly choose sine or cosine
        if np.random.rand() > 0.5:
            seq = [amp * np.sin(freq * t + phase) for t in range(sequence_length + 1)]
        else:
            seq = [amp * np.cos(freq * t + phase) for t in range(sequence_length + 1)]
            
        X.append(seq[:-1])
        y.append(seq[-1])
        
    X = np.array(X).reshape(num_sequences, sequence_length, 1)
    y = np.array(y)
    return X, y

def main():
    print("Generating data...")
    X, y = generate_data(1000, 50)
    
    # Split into train and test
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    print("Building LSTM model...")
    # Apply LSTM with 100 hidden units
    model = Sequential([
        LSTM(100, activation='tanh', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    print("Training model...")
    # Max 10 epochs
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    print("Evaluating model...")
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    
    # Predict a tested wave
    predictions = model.predict(X_test)
    
    # Optional: Plot the results for the first test sequence
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(X_test[0])), X_test[0], label='Input Sequence')
    plt.scatter([len(X_test[0])], [y_test[0]], color='green', label='True Next Value')
    plt.scatter([len(X_test[0])], [predictions[0]], color='red', label='Predicted Next Value')
    plt.title('LSTM Wave Prediction')
    plt.legend()
    plt.savefig('wave_prediction.png')
    print("Saved plot to wave_prediction.png")

if __name__ == "__main__":
    main()
