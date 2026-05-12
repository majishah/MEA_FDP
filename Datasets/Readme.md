# 📚 Machine Learning Foundations — FDP Workshop Resources

### Simple Downloadable Datasets for Faculty Development Program

This repository contains all the resources, dataset links, and guidance for the **Machine Learning Foundations** FDP workshop. Each dataset represents one type of data commonly used in AI, Data Science, and Machine Learning — explained through a **human learning lens**.

---

## 🎯 Theme of First Day(18-05-2026

> *"Machines learn the way humans do — from experience. They just do it faster."*

| What Humans Use | What Machines Use | Dataset Type |
|----------------|------------------|-------------|
| 🧠 Brain calculations | Spreadsheet / ML model | Numerical |
| 📖 Reading & language | NLP processing | Text |
| 🤚 Touch & balance | IoT sensors | Sensor |
| 👂 Hearing | Speech recognition | Voice |
| 👁️ Eyes | Computer vision | Image |
| 👀 Observation | Video analytics | Video |
| 🧭 Navigation sense | GPS tracking systems | GPS |

---

## 📦 Datasets — Download Links & Instructions

### 1. 🔢 Numerical Dataset — Iris Flower Dataset

| Property | Details |
|----------|---------|
| **Type** | Numerical / Tabular |
| **Use Case** | Classification |
| **Features** | Sepal length, sepal width, petal length, petal width |
| **Samples** | 150 |
| **Real-life analogy** | Like classifying students into categories based on their marks |

**🔗 Download:** [https://archive.ics.uci.edu/dataset/53/iris](https://archive.ics.uci.edu/dataset/53/iris)

**How to download:**
1. Click the link above → Click the **"Download"** button on the page
2. Extract the `.zip` file → You'll find `iris.data` (CSV format)
3. Open in Excel or load in Python:
```python
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, names=columns)
df.head()
```

**Workshop Activity:** Compare flower measurements and predict species.

---

### 2. 💬 Text Dataset — SMS Spam Collection

| Property | Details |
|----------|---------|
| **Type** | Text Data |
| **Use Case** | NLP / Spam Detection |
| **Features** | SMS text + label (spam or ham) |
| **Samples** | 5,574 messages |
| **Real-life analogy** | Like your brain filtering spam WhatsApp messages from real ones |

**🔗 Download:** [https://archive.ics.uci.edu/dataset/228/sms+spam+collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

**How to download:**
1. Click the link → Click **"Download"**
2. Extract the `.zip` → Open `SMSSpamCollection` file
3. Load in Python:
```python
import pandas as pd
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df.head()
```

**Workshop Activity:** Identify spam vs normal messages, word frequency analysis.

---

### 3. 📡 Sensor Dataset — Human Activity Recognition (Smartphones)

| Property | Details |
|----------|---------|
| **Type** | Sensor Data (Accelerometer + Gyroscope) |
| **Use Case** | Activity Recognition |
| **Activities** | Walking, sitting, standing, lying, walking upstairs/downstairs |
| **Samples** | 10,299 |
| **Real-life analogy** | Like how your smartwatch knows if you're walking or sleeping |

**🔗 Download:** [https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

**How to download:**
1. Click the link → Click **"Download"**
2. Extract the `.zip` → Navigate to the `UCI HAR Dataset` folder
3. Key files: `X_train.txt` (features), `y_train.txt` (labels), `features.txt` (column names)
4. Quick load in Python:
```python
import pandas as pd
features = pd.read_csv('UCI HAR Dataset/features.txt', sep=' ', names=['id', 'name'])
X_train = pd.read_csv('UCI HAR Dataset/train/X_train.txt', sep=r'\s+', header=None)
y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', header=None, names=['activity'])
print(f"Features: {X_train.shape[1]}, Samples: {X_train.shape[0]}")
```

**Workshop Activity:** Detect walking, sitting, standing — explain IoT and wearables.

---

### 4. 🎤 Voice Dataset — Free Spoken Digit Dataset (FSDD)

| Property | Details |
|----------|---------|
| **Type** | Audio / Voice Data |
| **Use Case** | Speech Recognition |
| **Contains** | Spoken digits 0–9 in `.wav` format |
| **Speakers** | 6 speakers, 3000 recordings |
| **Real-life analogy** | Like how Alexa / Google Assistant recognizes your voice commands |

**🔗 Download:** [https://github.com/Jakobovski/free-spoken-digit-dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)

**How to download:**
1. Click the link → Click the green **"Code"** button → **"Download ZIP"**
2. Extract → Audio files are in the `recordings/` folder
3. File naming format: `{digit}_{speaker}_{index}.wav` (e.g., `7_jackson_0.wav`)
4. Play and visualize in Python:
```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

audio, sr = librosa.load('recordings/7_jackson_0.wav')
plt.figure(figsize=(10, 3))
librosa.display.waveshow(audio, sr=sr)
plt.title('Waveform of Spoken Digit "7"')
plt.show()
```
> **Note:** Install librosa first — `pip install librosa`

**Workshop Activity:** Play audio samples, recognize spoken numbers.

---

### 5. 🖼️ Image Dataset — MNIST Handwritten Digits

| Property | Details |
|----------|---------|
| **Type** | Image Data (28×28 grayscale) |
| **Use Case** | Computer Vision / Image Classification |
| **Contains** | Handwritten digits 0–9 |
| **Samples** | 70,000 images (60K train + 10K test) |
| **Real-life analogy** | Like how your eyes recognize handwriting on a whiteboard |

**🔗 Download (Kaggle):** [https://www.kaggle.com/datasets/hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

**Easier alternative — load directly in Python (no download needed):**
```python
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
# Show a sample digit
plt.imshow(mnist.data[0].reshape(28, 28), cmap='gray')
plt.title(f'Label: {mnist.target[0]}')
plt.axis('off')
plt.show()
```

**Workshop Activity:** Show handwritten digits, explain image recognition basics.

---

### 6. 🎬 Video Dataset — UCF101 Action Recognition

| Property | Details |
|----------|---------|
| **Type** | Video Data |
| **Use Case** | Human Action Recognition |
| **Contains** | 101 action categories (sports, daily activities) |
| **Samples** | 13,320 video clips |
| **Real-life analogy** | Like how your brain recognizes actions when watching CCTV footage |

**🔗 Download:** [https://www.crcv.ucf.edu/data/UCF101.php](https://www.crcv.ucf.edu/data/UCF101.php)

**How to download:**
1. Click the link → Click the **"UCF101.rar"** download link
2. File is large (~6.5 GB) — **for the workshop, download only a few sample videos**
3. Extract using WinRAR or 7-Zip → Videos are organized in folders by action category
4. For a quick demo, pick 2–3 folders like `Basketball/`, `Walking/`, `Typing/`

> **⚠️ Workshop Tip:** Don't download the full dataset during the session. Just download 2–3 sample videos beforehand and play them to explain the concept. The goal is to show what video data looks like, not to train a model.

**Workshop Activity:** Identify actions like running, jumping — explain video analytics.

---

### 7. 📍 GPS Dataset — GeoLife GPS Trajectories

| Property | Details |
|----------|---------|
| **Type** | GPS / Location Data |
| **Use Case** | Mobility & Trajectory Analysis |
| **Contains** | GPS trajectories from 182 users over 3+ years |
| **Format** | `.plt` files with latitude, longitude, altitude, timestamp |
| **Real-life analogy** | Like how Google Maps tracks your route and predicts traffic |

**🔗 Download:** [https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)

**How to download:**
1. Click the link → Scroll down to **"Download"** section
2. Download the `.zip` file (~300 MB)
3. Extract → Each user has a folder with `.plt` trajectory files
4. Load one trajectory in Python:
```python
import pandas as pd

# Load a single trajectory file
cols = ['lat', 'lon', 'zero', 'altitude', 'days', 'date', 'time']
df = pd.read_csv('Data/000/Trajectory/20081023025304.plt',
                  skiprows=6, names=cols)
print(df[['lat', 'lon', 'altitude', 'date', 'time']].head())
```

> **💡 Workshop Tip:** Pick just one user's folder (e.g., `Data/000/`) for the demo.

**Workshop Activity:** Plot routes on a map, explain navigation and tracking.


## 🔧 Quick Setup for Participants

**Option A: Google Colab (Recommended — No Installation Needed)**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Click **"New Notebook"**
4. Copy-paste the Python code snippets from the sections above
5. Click the ▶️ play button to run each cell

**Option B: Local Setup**

Install these Python libraries:
```bash
pip install numpy pandas matplotlib scikit-learn librosa
```


## 📝 License

This repository is for **educational purposes** as part of the Faculty Development Program. All datasets belong to their respective creators and are publicly available for research and education.
