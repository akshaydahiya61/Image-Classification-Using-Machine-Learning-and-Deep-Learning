# Image-Classification-Using-Machine-Learning-and-Deep-Learning
This project was developed as part of my coursework to explore machine learning and deep learning techniques for image classification tasks using the CIFAR-10 dataset.

The project evaluates the performance of three different approaches:

Traditional Machine Learning (ML) model using Support Vector Machine

Convolutional Neural Network (CNN) model

Fully Connected Neural Network (NN) model

📁 Project Structure
Big Data CW Final.ipynb — Jupyter notebook containing complete code for data preprocessing, model building, training, evaluation, and result analysis.

CIFAR-10 dataset loaded via Keras Datasets.

📌 Methods and Workflow
📊 1️⃣ Support Vector Machine
The CIFAR-10 images were flattened into 1D vectors.

Applied a Support Vector Machine.

Evaluated using accuracy score, confusion matrix, and classification report.

Result:
Approximate accuracy: 31%

🧠 2️⃣ Convolutional Neural Network (CNN)
CNN architecture:

2 Conv2D layers with ReLU activation + MaxPooling2D

Flatten layer

Dense layers: 128 neurons (ReLU) → 10 neurons (Softmax)

Compiled with categorical crossentropy loss and Adam optimizer.

Trained over multiple epochs with data normalization.

Result:
Approximate accuracy: 48% after training.

🖥️ 3️⃣ Fully Connected Neural Network (NN)
Flattened images into 3072-length vectors.

Architecture:

Dense (512, ReLU) → Dropout (0.5)

Dense (256, ReLU) → Dropout (0.3)

Dense (10, Softmax)

Compiled and trained similarly to the CNN.

Result:
Approximate accuracy: 47%

📈 Performance Summary

Model	Test Accuracy
Support Vector Machine	~31%
CNN	~48%
Fully Connected NN	~47%
📚 Technologies Used
Python

Jupyter Notebook

TensorFlow / Keras

Scikit-learn

Matplotlib, Seaborn for visualization

📄 How to Run
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook "Image-Classification-Using-Machine-Learning-and-Deep-Learning.ipynb"
✨ Key Learning Outcomes
Handling image data in traditional and deep learning pipelines.

Implementing and tuning CNN and Dense Neural Networks.

Evaluating model performance using confusion matrices and classification reports.

Comparing ML and DL approaches for image classification.
