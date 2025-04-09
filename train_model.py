import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

images = np.load("images.npy")
labels = np.load("labels.npy")

num_samples = images.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)
split_index = int(0.9 * num_samples)
train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train, X_test = images[train_indices], images[test_indices]
y_train, y_test = labels[train_indices], labels[test_indices]

X_train = np.expand_dims(X_train, axis=-1) / 255.0
X_test = np.expand_dims(X_test, axis=-1) / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(26, activation='softmax')  
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),  
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

model.save("letter_recognition_model.keras")

def verify_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    correct_count = 0

    for i in range(len(X_test)):
        true_label = int(y_test[i]) 
        predicted_label = predicted_labels[i]
        print(f"True Label: {chr(true_label + 65)}, Predicted Label: {chr(predicted_label + 65)}")
        if true_label == predicted_label:
            correct_count += 1

    accuracy = correct_count / len(X_test)
    print(f"Manual verification accuracy: {accuracy}")

verify_predictions(model, X_test, y_test)
