<a id="readme-top"></a>

<!-- ABOUT THE PROJECT -->

## About The Project

<!-- ![Product Name Screen Shot][product-screenshot]<br> -->

Cars_VQA is a Visual Question Answering (VQA) project focused on answering questions about cars in images. It provides a dataset of car images along with questions and answers, and includes models trained to perform VQA on this dataset. The project aims to advance research in VQA for specific domains, particularly automotive-related applications.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
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

### Built With

- [![PyTorch][PyTorch]][PyTorch-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- Python 3.7 or higher
  ```sh
  pip install --upgrade pip
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/khoadole/Cars_VQA.git
   ```
2. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Crawling and Preprocessing Data

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

### Training and Testing Models

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

[product-screenshot]: images/screenshot.png
[PyTorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
