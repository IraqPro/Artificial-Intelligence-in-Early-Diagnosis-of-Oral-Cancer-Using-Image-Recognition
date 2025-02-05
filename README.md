
# Artificial-Intelligence-in-Early-Diagnosis-of-Oral-Cancer-Using-Image-Recognition

A Graduation research by **Ali Muhannad Ahmed** for University of Karbala, College of Dentistry.

## Overview

Oral cancer is the sixth most prevalent cancer worldwide and accounts for a significant proportion of head and neck malignancies. This project leverages Artificial Intelligence (AI) and Machine Learning, specifically Deep Learning, to detect and classify oral cancer using medical image recognition techniques. By automating the analysis of clinical images, this AI model offers a promising solution to aid in early diagnosis and improve patient outcomes.

## Project Requirements

This project is built in Python and requires several libraries for training, testing, and deploying the model. Below are the steps to install and run the project.

## Installation and Setup

### 1. **Clone the Repository**

First, clone the project repository to your local machine:

```bash
git clone [https://github.com/your-username/Artificial-Intelligence-in-Early-Diagnosis-of-Oral-Cancer-Using-Image-Recognition.git](https://github.com/IraqPro/Artificial-Intelligence-in-Early-Diagnosis-of-Oral-Cancer-Using-Image-Recognition/tree/master)
cd Artificial-Intelligence-in-Early-Diagnosis-of-Oral-Cancer-Using-Image-Recognition
```

### 2. **Install via Conda**

#### For Linux, macOS, and Windows:

If you have **Conda** installed (via Anaconda or Miniconda), you can easily set up an environment for this project. Follow these steps:

1. **Create a Conda Environment**  
   Create a new Conda environment for the project, specifying the Python version (e.g., Python 3.8):

   ```bash
   conda create --name oral-cancer-ai python=3.8
   ```

2. **Activate the Environment**  
   Once the environment is created, activate it:

   For **Linux/macOS**:

   ```bash
   conda activate oral-cancer-ai
   ```

   For **Windows**:

   ```bash
   activate oral-cancer-ai
   ```

3. **Install Dependencies**  
   After activating the environment, install the required dependencies. You can either manually install the necessary libraries or use the `requirements.txt` file (if available).

   **Manually install dependencies:**
   - For **TensorFlow** (recommended for deep learning tasks):
     ```bash
     conda install tensorflow
     ```
   - For **PyTorch** (optional, if you prefer using PyTorch):
     ```bash
     conda install pytorch torchvision torchaudio -c pytorch
     ```
   - Additional dependencies:
     ```bash
     conda install numpy pandas opencv matplotlib scikit-learn Pillow
     ```

4. **Download Dataset**

   This project requires access to the oral cancer image dataset. Ensure the dataset is placed in the following directory structure (relative to the project root):

   ```
   datasets/
       OralCancer/
           train/
           test/
           train.csv
           submission.csv
   ```

   You can add the dataset manually or modify the dataset path in the code if necessary.

5. **Run the Model**

   After setting up the environment and dataset, you can start training the model. To train the model, use the following command:

   ```bash
   python OralCancer.py
   ```

   This will start the training process. The model will be saved in the `models/` directory after training.

6. **Model Inference**

   Once the model is trained, you can run inference on new images using web app,copy the name of the ckpt file and edit it in app.py then run:

   ```bash
   python app.py
   ```

   It will give you a localhost url for prediction.

## Project Structure

- `OralCancer.py`: Script to train the AI model.
- `app.py`: Script for web interface interaction.
- `datasets/`: Folder to store the training and testing datasets.
- `models/`: Folder where the trained model is saved.
- `README.md`: This file!

## Required Libraries

- **Python 3.7+**
- TensorFlow/PyTorch (depending on your preference)
- NumPy
- OpenCV
- pandas
- scikit-learn
- matplotlib
- Pillow

You can install the necessary libraries with the following command:

```bash
pip install tensorflow numpy opencv-python pandas scikit-learn matplotlib Pillow
```

Or, if you’re using PyTorch:

```bash
pip install torch torchvision numpy opencv-python pandas scikit-learn matplotlib Pillow
```

## Contributing

If you wish to contribute to this project, feel free to fork the repository, submit issues, and create pull requests. Contributions, suggestions, and improvements are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
