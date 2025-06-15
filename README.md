<a id="readme-top"></a>

<!-- ABOUT THE PROJECT -->

## 📘 About The Project

![Product Name Screen Shot][product-screenshot]<br>

Cars_VQA is a Visual Question Answering (VQA) project focused on answering questions about cars in images. It provides a dataset of car images along with questions and answers, and includes models trained to perform VQA on this dataset. The project aims to advance research in VQA for specific domains, particularly automotive-related applications.<br>

## 🌐 Live Demo

The project has been successfully deployed online.
🔗 You can try it here:  
👉 [https://carsvqa-production.up.railway.app](https://carsvqa-production.up.railway.app)

## 🗂️ Dataset

- Crawl from : https://www.cars.com/new-cars/ <br>
- Upload to Hugging Face : https://huggingface.co/datasets/khoadole/cars_8k_balance_dataset_full_augmented_v2

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#framework--libraries">Framework & Libraries</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>

### 🧰 Framework & Libraries

- [![PyTorch][PyTorch]][PyTorch-url]
- [![NumPy][NumPy]][NumPy-url]
- [![Pandas][Pandas]][Pandas-url]
- [![Scikit-learn][Scikit-learn]][Scikit-learn-url]
- [![Ultralytics][Ultralytics]][Ultralytics-url]

### Deployment & Optimization

- [![ONNX][ONNX]][ONNX-url]
- [![Docker][Docker]][Docker-url]

### Data & Visualization

- [![Matplotlib][Matplotlib]][Matplotlib-url]
- [![Seaborn][Seaborn]][Seaborn-url]

### Web Framework

- [![Flask][Flask]][Flask-url]
- [![HTML5][HTML5]][HTML5-url]
- [![Tailwind CSS][TailwindCSS]][TailwindCSS-url]
- [![JavaScript][JavaScript]][JavaScript-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<!-- GETTING STARTED -->

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### 📦 Prerequisites

- Python 3.7 or higher
  ```sh
  pip install --upgrade pip
  ```

### 🛠️ Installation

1. Clone the repo
   ```sh
   git clone https://github.com/khoadole/Cars_VQA.git
   ```
2. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## ▶️ Usage

### 🕸️ Crawling and Preprocessing Data

1. **Crawl Data**: Use the Jupyter Notebook to crawl data for the project.

   ```sh
   cd crawl_data
   jupyter notebook crawl_data.ipynb
   ```

   Follow the instructions in `crawl_data.ipynb` to collect the dataset.

2. **Preprocess Raw Data**: Preprocess the crawled data to clean and format it.

   ```sh
   jupyter notebook preprocess_raw_data.ipynb
   ```

   Execute the cells in `preprocess_raw_data.ipynb` to prepare the raw data.

3. **Prepare Dataset for Training**: Finalize the dataset for model training.
   ```sh
   jupyter notebook preprocess_dataset.ipynb
   ```
   Run `preprocess_dataset.ipynb` to generate the final dataset ready for training.

### 🧠 Training and Testing Models

1. Change to the model directory:

   ```sh
   cd src/models/<model_name>
   ```

   Replace `<model_name>` with the specific model name (e.g., `BOW`).

2. Train the model:

   ```sh
   python train.py
   ```

3. Test the model on the test dataset:

   ```sh
   python test.py
   ```

4. Results will be saved in the directory `models/<trained_model_name>/`, containing:
   - `log`: Training and evaluation logs
   - `<model_name>.pth`: Trained model weights in checkpoint folder
   - Vocabulary file for the dataset

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[product-screenshot]: images/landingpage.png
[PyTorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[Flask]: https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/
[HTML5]: https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white
[HTML5-url]: https://developer.mozilla.org/en-US/docs/Web/HTML
[TailwindCSS]: https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white
[TailwindCSS-url]: https://tailwindcss.com/
[JavaScript]: https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E
[JavaScript-url]: https://developer.mozilla.org/en-US/docs/Web/JavaScript
[NumPy]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Pandas]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[Scikit-learn]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[Scikit-learn-url]: https://scikit-learn.org/
[ONNX]: https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white
[ONNX-url]: https://onnx.ai/
[Matplotlib]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org/
[Seaborn]: https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white
[Seaborn-url]: https://seaborn.pydata.org/
[Ultralytics]: https://img.shields.io/badge/Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black
[Ultralytics-url]: https://ultralytics.com/
[Docker]: https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://www.docker.com/
