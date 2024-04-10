# CS4120-Project
## By: David Pogrebitskiy and Jacob Ostapenko

### Project Description
This project aims to compare the performance of different classification techniques with different types of embeddings. The embeddings used are Doc2Vec and BERT. The classification techniques used are Logistic Regression, Random Forest, Support Vector Machine, Naive Bayes, K-Nearest Neighbors, and Feed-Forward Neural Network. The dataset used was the [All the news](https://www.kaggle.com/datasets/snapcrack/all-the-news?resource=download) from Kaggle. It contains news articles from different sources and their corresponding authors. We filtered down the dataset to include 20 most published authors for author attribution. The models were trained on the embeddings of the news articles and tested on the authors. The performance of the models was evaluated using accuracy, precision, recall, and F1 score.

### Project Structure
The project is structured as follows:
- `evaluation/`: Contains the evaluation results of the models.
- `data/`: Contains the dataset used for the project. The raw dataset is too large to be included in the repository. The filtered dataset is serialized as `data/cleaned_articles.pkl`.
- `models/`: Contains the classes used to modularize the models.
- `pickles/`: Contains the pickled objects used to store label encoders.
- `preprocessing/`: Contains the notebooks used to preprocess the data and get the features.
- `utils.py`: Contains the utility functions used in the project.
- `Embeddings.py`: Contains the classes used to store the embeddings of the dataset

### References
- Abbasi, Ahmed, et al. "Authorship Identification Using Ensemble Learning." Scientific Reports, vol. 12, no. 1, 2022, pp. 1-16, https://doi.org/10.1038/s41598-022-13690-4. Accessed 10 Apr. 2024.