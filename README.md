
# 🧠 Multi-Cancer Image Classification

This project aims to classify different types of cancer images using a **deep learning Convolutional Neural Network (CNN)**. The model is built with **TensorFlow/Keras**.

## 📂 Project Structure

* Data preprocessing is done with `ImageDataGenerator` (normalization and train/validation split).
* The CNN model is created with **Conv2D**, **MaxPooling2D**, and **Dense** layers.
* After training, the model is saved as `.h5`.
* The function `gercek_deger()` allows testing predictions on a single image.

## ⚙️ Dependencies

* **TensorFlow / Keras**
* **NumPy**
* **Matplotlib**

## 🚀 Model Architecture

1. **Conv2D + MaxPooling2D** → Feature extraction from images
2. **Conv2D + MaxPooling2D**
3. **Conv2D + MaxPooling2D**
4. **Flatten** → Convert data into a vector
5. **Dense (512, ReLU)** → Fully connected layer
6. **Dense (Softmax)** → Output layer with class probabilities

## 🏋️ Training

* Input image size: **150x150**
* Batch size: **32**
* Optimizer: **Adam**
* Loss: **Categorical Crossentropy**
* Metrics: **Accuracy**
* Epochs: **10**

## 📊 Usage

### Train the model

```python
model.fit(train_generator, validation_data=validation_generator, epochs=10)
```

### Save the model

```python
model.save("image_classifier.h5")
```

### Predict on a new image

```python
gercek_deger("test_image.jpg", model, train_generator.class_indices)
```

---

### 📈 Training Curves




