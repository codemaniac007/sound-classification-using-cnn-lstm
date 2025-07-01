
# 🔊 Environmental Sound Classification using CNN-LSTM

This project implements an **Environmental Sound Classification (ESC)** system using a hybrid **Convolutional Neural Network (CNN)** and **Long Short-Term Memory (LSTM)** model. It is trained and evaluated on the **UrbanSound8K** dataset, which contains real-world urban sound samples labeled into 10 distinct categories.

---

## 📁 Dataset: UrbanSound8K

- 🎵 **Classes** (10):  
  - `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`,  
    `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`

- 🗂 **Total Samples**: 8,732 audio clips  
- ⏱ **Sampling Rate**: 22,050 Hz  
- 🌐 [Download Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)

---

## 🧠 Model Architecture

The architecture combines CNN’s spatial feature extraction with LSTM’s sequential modeling power.

```text
Input → Mel-Spectrogram → CNN Layers → Reshape → LSTM Layers → Dense Layers → Output (Softmax)
````

### 📌 Components:

* **Mel Spectrogram**: 128 Mel bands per clip.
* **CNN Layers**: Convolution + MaxPooling + Dropout.
* **LSTM Layer**: Processes time-series features extracted by CNN.
* **Dense Layers**: Classifier with softmax activation.

---

## 🛠 Features & Techniques

* 🎧 Mel-spectrograms as input features.
* 🧼 Preprocessing: Audio resampling, mono-channel conversion, normalization.
* 🔁 K-fold cross-validation using fold metadata in UrbanSound8K.
* 📊 Evaluation: Accuracy, Confusion Matrix, Classification Report.

---

## 💻 Requirements

Install dependencies using pip:

```bash
pip install tensorflow librosa numpy matplotlib scikit-learn
```

---

## 🚀 How to Run

1. **Clone this repository**:

   ```bash
   git clone https://github.com/yourusername/esc-cnn-lstm.git
   cd esc-cnn-lstm
   ```

2. **Download UrbanSound8K Dataset**:

   * Extract it into a folder named `UrbanSound8K` in the project root.
   * Update the dataset path in the notebook if necessary.

3. **Run the Notebook**:

   * Open `urbansound8k-cnn-lstm-submission.ipynb` using Jupyter Notebook or Google Colab.
   * Execute all cells sequentially.

---

## 📊 Results

| Metric                     | Score  |
| -------------------------- | ------ |
| ✅ **Train Accuracy**       | 0.9928 |
| 🧪 **Validation Accuracy** | 0.8725 |
| 🧾 **Test Accuracy**       | 0.9332 |

* 📉 Confusion Matrix:

  * Evaluates class-wise performance.
* 📃 Classification Report:

  * Contains precision, recall, and F1-score for each class.

---

## 📌 Future Improvements

* 🎛 Add data augmentation (e.g., time-stretching, pitch-shifting).
* 🧠 Integrate attention or Transformer layers.
* 📦 Experiment with more advanced models (CRNN, ResNet).
* 🧪 Hyperparameter optimization using Optuna/KerasTuner.

---

## 📚 References

* 📄 [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)
* 🎧 [Librosa Documentation](https://librosa.org/)
* 🔧 [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
* 📘 [Keras Docs](https://keras.io/)

```
```
