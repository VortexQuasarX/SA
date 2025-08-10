# Shilling Attack Detection in YouTube Comments using Hybrid Deep Learning

This project focuses on detecting various types of shilling attacks (spam comments) in YouTube comment sections. It employs a hybrid deep learning model that combines BERT with Convolutional Neural Networks (CNNs) and compares its performance against other transformer-based models.

## Data

The project utilizes the following datasets:

1.  **`youtube_comments_oneplus copy (1).csv`**: This is the initial raw dataset of YouTube comments. (Note: This file is assumed to be present but is not included in this repository).
2.  **`spam_detected.csv`**: This dataset is generated from the raw comments by:
    *   Extracting text-based features (length, word count, sentiment, subjectivity, personal pronoun ratio, repetitive phrases).
    *   Extracting behavioral features (time since last review, review burst, duplicate review).
    *   Extracting reviewer-based features (review count per user, account age, review diversity).
    *   Calculating a `spam_score` based on these features and classifying comments as spam if the score exceeds a defined threshold.
3.  **`spam_detected_modify.csv`**: This dataset is an augmented version of `spam_detected.csv`. It includes synthetically generated shilling attacks to create a more robust training and evaluation set. The types of attacks injected include:
    *   Shilling attacks (promoting products/brands)
    *   Promotional attacks (promoting other channels/content)
    *   Discrediting attacks (undermining the video/creator)
    *   Astroturfing attacks (coordinated messages from multiple fake accounts)
    *   Clickbait attacks (sensational headlines with misleading links)
    *   Emotional manipulation attacks (playing on user emotions)
    These attacks are generated using the `Faker` library.

## Models

The primary model and other experimented models are:

1.  **Hybrid BERT + CNN**:
    *   This is the main model used for spam detection.
    *   It utilizes a pre-trained BERT model (`bert-base-uncased`) for sequence classification to get text embeddings.
    *   These embeddings are then passed through two 1D Convolutional Neural Network (CNN) layers (`Conv1d`) with ReLU activation, followed by global average pooling.
    *   A final fully connected layer outputs the classification (spam/not spam).

2.  **Other Transformer-based Models (for comparison)**:
    *   **Regular BERT**: A standard BERT classifier without the additional CNN layers.
    *   **RoBERTa-CNN**: Hybrid model using RoBERTa (`roberta-base`) as the transformer backbone with CNN layers.
    *   **DistilBERT-CNN**: Hybrid model using DistilBERT (`distilbert-base-uncased`) as the transformer backbone with CNN layers.
    *   **ALBERT-CNN**: Hybrid model using ALBERT (`albert-base-v2`) as the transformer backbone with CNN layers.

## Methodology

1.  **Initial Spam Detection (Rule-based)**:
    *   The `youtube_comments_oneplus copy (1).csv` dataset is processed.
    *   Features related to comment text, user behavior, and reviewer history are engineered.
    *   A `spam_score` is assigned based on a set of predefined rules (e.g., short review length, high sentiment, low unique word ratio).
    *   Comments exceeding a `SPAM_THRESHOLD` are labeled as spam. This output is `spam_detected.csv`.

2.  **Shilling Attack Generation & Dataset Augmentation**:
    *   The `spam_detected.csv` is augmented with various synthetically generated shilling attacks using the `Faker` library to create `spam_detected_modify.csv`. This helps in training models that are more robust to diverse spam techniques.

3.  **Model Training and Evaluation**:
    *   **Tokenization**: YouTube comments (text) are tokenized using `BertTokenizer` from the Transformers library (`bert-base-uncased`), with padding and truncation to a max length of 128.
    *   **Training**:
        *   The Hybrid BERT+CNN model is trained for 10 epochs on both `spam_detected.csv` and `spam_detected_modify.csv`.
        *   Comparative models (Regular BERT, RoBERTa-CNN, DistilBERT-CNN, ALBERT-CNN) are trained for 3 epochs on `spam_detected_modify.csv`.
        *   AdamW optimizer and CrossEntropyLoss are used.
    *   **Evaluation**: Models are evaluated based on:
        *   Accuracy
        *   Precision
        *   Recall
        *   F1-score
        *   Classification Report
        *   Confusion Matrix

## Results

*   **Hybrid BERT+CNN on `spam_detected.csv`**: Achieved an accuracy of approximately 96.43%. The model showed good precision for non-spam but had a lower recall for spam comments, indicating it missed some spam instances.
*   **Hybrid BERT+CNN on `spam_detected_modify.csv` (with shilling attacks)**: Achieved an accuracy of approximately 95.74%. The inclusion of diverse shilling attacks helped improve the model's ability to correctly identify spam (higher recall for spam class) while maintaining high precision for non-spam.
*   **Comparative Models**:
    *   The Hybrid BERT+CNN generally outperformed the regular BERT model and other hybrid transformer variants (RoBERTa-CNN, DistilBERT-CNN, ALBERT-CNN) when trained on the augmented dataset (`spam_detected_modify.csv`) for 3 epochs.
    *   BERT-CNN showed the best balance of precision and recall among the compared models.
    *   DistilBERT-CNN and ALBERT-CNN had difficulty identifying spam comments in the 3-epoch training setup.
    *   The notebook includes detailed classification reports and plots comparing these models.

## How to Run

1.  **Dependencies**:
    *   pandas
    *   numpy
    *   torch
    *   transformers (specifically `BertTokenizer`, `BertForSequenceClassification`, `BertModel`, etc.)
    *   scikit-learn (`accuracy_score`, `classification_report`, `train_test_split`, `confusion_matrix`)
    *   textblob
    *   faker
    *   matplotlib
    *   seaborn
    *   tqdm

    You can install them using pip:
    ```bash
    pip install pandas numpy torch transformers scikit-learn textblob faker matplotlib seaborn tqdm
    ```

2.  **Notebook**: All the code for data processing, model definition, training, and evaluation is contained within the Jupyter Notebook: `Hybrid_+_cnn_social_media_ (1).ipynb`.

3.  **Input Data**:
    *   Ensure the initial dataset `youtube_comments_oneplus copy (1).csv` is in the same directory as the notebook or update the path in the notebook accordingly.
    *   The notebook will generate `spam_detected.csv` and `spam_detected_modify.csv`.

4.  **Execution**: Open and run the cells in the Jupyter Notebook sequentially.

## File Structure

```
.
├── Hybrid_+_cnn_social_media_ (1).ipynb  # Main Jupyter Notebook
├── youtube_comments_oneplus copy (1).csv # Initial raw dataset (assumed, not in repo)
├── spam_detected.csv                     # Intermediate dataset after rule-based spam detection
├── spam_detected_modify.csv              # Augmented dataset with injected shilling attacks
└── README.md                               # This file
```

## Future Work
* Explore more sophisticated data augmentation techniques.
* Fine-tune hyperparameters for each model.
* Experiment with larger and more diverse datasets.
* Deploy the best performing model as an API for real-time spam detection.
