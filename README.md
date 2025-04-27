
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

## 📝 How to Run Locally

To run the app locally, follow these steps:

### 1. Clone the repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/traffic-sign-classifier.git
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

You can find the **GTSRB - German Traffic Sign Recognition** dataset at [this link](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

After downloading the dataset, make sure the directory structure looks like this:

```
traffic-sign-classifier/
│
├── archive/
│    ├── Train.csv
│    ├── Test.csv
│    ├── Images/  # All image files for training and testing
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
- `archive/`: Contains the dataset (Train.csv, Test.csv, and image folder).
- `model/`: Store the trained model file (`.pth`) here (optional).

## 🏆 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

Feel free to contribute, suggest improvements, or ask questions by opening an issue!
