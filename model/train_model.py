import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy

def create_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    return model

def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=BinaryCrossentropy(),
                  metrics=[Accuracy()])
    return model

def main():
    
    train_images = 'preprocessed_images.npy'
    train_labels = 'preprocessed_label.npy'
    
    # Load data
    X_train = np.load(train_images)
    y_train = np.load(train_labels)
    
    # Create and compile the model
    model = create_model()
    model = compile_model(model)
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Save the trained model
    model.save('vehicle_cutin_detection_model.h5')
    
    print("Model training completed and saved for vehicle cutin system.")

if __name__ == '__main__':
    main()
