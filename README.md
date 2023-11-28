# Sign Language Translator Project 🤟

## Table of Contents
- [About the Project](#about-the-project) 📖
- [Project Files Overview](#project-files-overview) 🗂️
  - [Code Files](#code-files) 💻
- [Data](#data)💾
- [Getting Started](#getting-started) 🚀
- [Contributing](#contributing) ✨

## About the Project 📖
The Sign Language Translator is an innovative project aimed at bridging the communication gap between deaf individuals and those unfamiliar with sign language. Utilizing advanced computer vision and natural language processing techniques, this project translates sign language from videos into written text. This breakthrough tool promises to enhance communication and accessibility for deaf individuals in a variety of settings.

## Project Files Overview 🗂️

### Code Files 💻

1. **requirements.txt**
   - Description: requirements

2. **data.py**
   - Description: py script for dataset loading

3. **classes.py**
   - Description: dictionary with all classes

4. **data_preprocessing.ipynb**
   - Description: Code for preprocessing the data from the "Slovo - Russian Sign Language Dataset". Includes steps for data cleaning, normalization.

5. **model_analiitic_written_models.ipynb**
   - Description: Contains various models experiments conducted during the project. Includes different hand-written models for sign language recognition and comparative analysis of treir performance.

6. **model_analitic_pretrained_model.ipynb**
   - Description: MViT16-4 model implementation notebook. Includes the preprocessing, training, and evaluation.

7. **training_preproc_written_model.ipynb**
   - Description: TwoStream3DConvNet model preprocessing.

8. **hand-written-model.ipynb**
   - Description: Training and evaluation of TwoStream3DConvNet model.


## Data 💾
[Slovo](https://www.kaggle.com/datasets/kapitanov/slovo/data): video dataset for Russian Sign Language (RSL) recognition

## Getting Started 🚀

To get started with the Sign Language Translator project, follow these steps:

### Prerequisites
Before running the project, ensure that you have the following installed:
- Python 3.7 or higher
- Jupyter Notebook or Jupyter Lab

### Installation

1. **Clone the repository**:

```bash
git clone https://your-repository-link-here.git
cd sign-language-translator
```

2. **Create a virtual environment** (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
```


3. **Install the required packages**:

```bash
pip install -r requirements.txt
```


4. **Load dataset**:

```bash
python data.py
```

## Contributing ✨

Thanks goes to these wonderful people:
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
   <td align="center" width = "185px"><a href="https://github.com/YarikaAA"><img src="https://avatars.githubusercontent.com/u/54471402?v=4" width="100px;" alt=""/ class="avatar"><br /> <b>Arina Yartseva</b></a><br /> <sub><b><a href="mailto:a.yartseva@innopolis.university" title="mail to Arina">📧 contact</a></b></sub><br /></td>
   <td align="center" width = "170px"><a href="https://github.com/veriFCKation"><img src="https://avatars.githubusercontent.com/u/99489584?v=4" width="100px;" alt=""/><br /><b>Ksenia Shchekina</b></a><br /> <sub><b><a href="mailto:k.shchekina@innopolis.university" title="mail to Ksenia">📧 contact</a></b></sub><br /></td>
   <td align="center" width = "185px"><a href="https://github.com/Poleeknow"><img src="https://avatars.githubusercontent.com/u/106336793?v=4" width="100px;" alt=""/><br /> <b>Polina Bazhenova</b></a><br /> <sub><b><a href="mailto:p.bazhenova@innopolis.university" title="mail to Polina">📧 contact</a></b></sub><br /></td>
</tr>
</table>

<!-- markdownlint-restore -->

<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
