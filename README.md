
# 🚦 Traffic Sign Classifier

A deep learning-based web application to classify German traffic signs using a trained ResNet18 model. The app allows users to upload an image, preview it, and predict the traffic sign class.

## 📸 One click recognition
- **Upload an image**
- **Instantly preview the uploaded image**
- **Classifies the traffic sign** and shows the predicted class name.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Deep Learning**: PyTorch (ResNet18 model)
- **Frontend**: HTML, CSS (Bootstrap), JavaScript (AJAX for better user experience)
- **Image Processing**: Pillow, TorchVision



### Preview Image

![Traffic Sign Classifier Preview](static/previews/img1.png)



## 📝 How to Run Locally

To run the app locally, follow these steps:

### 1. Clone the repository

First, clone the repository to your local machine:

```bash
git clone  https://github.com/Sam-Begin-tech/traffic-sign-classifier.git
cd traffic-sign-classifier
```

### 2. Install Python and Required Dependencies

Make sure you have Python installed. Then, create a virtual environment (optional but recommended):

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows**:
  ```bash
  .\venv\Scripts\activate
  ```

- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

Now, install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set Up Your Dataset

Make sure the dataset files (`Train.csv`, `Test.csv`, and images) are available in the `archive/` directory.

You can find the **GTSRB - German Traffic Sign Recognition** dataset at [this link](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data).

After downloading the dataset, make sure the directory structure looks like this:

```
traffic-sign-classifier/
│
│
├── app.py        # Flask backend code
├── templates/
│    └── index.html
└── static/
     └── css/
     └── js/
     └── (optional assets like images)
```

### 4. Start the Flask Application

Run the Flask application:

```bash
python app.py
```

### 5. Open the Web App

Once the server is running, open your browser and visit the following URL:

```
http://127.0.0.1:5000/
```

You should see the **Traffic Sign Classifier** app where you can upload images and get predictions in real-time.

## 🚀 Deployment Options

You can deploy this application on a cloud platform like **Render**, **Railway**, or **Heroku**. You might need to set up a few configurations for cloud deployment, but the basic process remains the same.

## 📈 Model Performance

- **Validation Accuracy**: 99.14%

The model was trained using the **ResNet18** architecture, which was fine-tuned on the **GTSRB - German Traffic Sign Recognition** dataset. The final model has achieved an accuracy of **99.14%** on the validation set.

## 🧑‍💻 Dataset

This project uses the **GTSRB - German Traffic Sign Recognition Benchmark** dataset. You can download it from the official site: [GTSRB Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The dataset includes images and metadata (CSV files) for different traffic sign classes.

## 💡 Features

- **Image Upload and Preview**: Users can upload an image, and the app will show a preview of the image along with the predicted traffic sign class.
- **Real-time Prediction**: The model uses a **ResNet18** architecture fine-tuned for traffic sign recognition.
- **Responsive Design**: The web interface is built with **Bootstrap**, ensuring a great user experience on any device.

## ⚙️ Directory Structure

- `app.py`: The Flask server code, where the app runs.
- `templates/`: Contains the HTML files used for the frontend of the app.
- `static/`: Contains static assets like CSS, JS, and images.
- `model/`: Store the trained model file (`.pth`) here (optional).




## Traffic Sign Classes

Here are the classes of traffic signs the model can classify:

- **0**: Speed limit (20km/h)
- **1**: Speed limit (30km/h)
- **2**: Speed limit (50km/h)
- **3**: Speed limit (60km/h)
- **4**: Speed limit (70km/h)
- **5**: Speed limit (80km/h)
- **6**: End of speed limit (80km/h)
- **7**: Speed limit (100km/h)
- **8**: Speed limit (120km/h)
- **9**: No passing
- **10**: No passing for vehicles over 3.5 metric tons
- **11**: Right-of-way at the next intersection
- **12**: Priority road
- **13**: Yield
- **14**: Stop
- **15**: No vehicles
- **16**: Vehicles over 3.5 metric tons prohibited
- **17**: No entry
- **18**: General caution
- **19**: Dangerous curve to the left
- **20**: Dangerous curve to the right
- **21**: Double curve
- **22**: Bumpy road
- **23**: Slippery road
- **24**: Road narrows on the right
- **25**: Road work
- **26**: Traffic signals
- **27**: Pedestrians
- **28**: Children crossing
- **29**: Bicycles crossing
- **30**: Beware of ice/snow
- **31**: Wild animals crossing
- **32**: End of all speed and passing limits
- **33**: Turn right ahead
- **34**: Turn left ahead
- **35**: Ahead only
- **36**: Go straight or right
- **37**: Go straight or left
- **38**: Keep right
- **39**: Keep left
- **40**: Roundabout mandatory
- **41**: End of no passing
- **42**: End of no passing by vehicles over 3.5 metric tons


## 🏆 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

---

Feel free to contribute, suggest improvements, or ask questions by opening an issue!
