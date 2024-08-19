# %% import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.utils import to_categorical

import pickle

# %% load data

# load data (cleaned with preprocess_data.ipynb)
with open('data/processed/isic_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

X_train            = loaded_data[0]
X_test             = loaded_data[1]
y_train            = loaded_data[2]
y_test             = loaded_data[3]
labels             = loaded_data[4]
metadata           = loaded_data[5]

# %% load model

# load classification model
with open('models/CNN_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# %% reshape data

X_train = X_train.reshape((9376, 450, 600, 3))
X_test  = X_test.reshape((2344, 450, 600, 3))


# %% predict



# 1. Evaluate the model on the test data
num_classes = len(np.unique(y_test))
# test_loss, test_accuracy = model.evaluate(X_test, to_categorical(y_test, num_classes), verbose=0)
# print(f'Test Loss: {test_loss:.4f}')
# print(f'Test Accuracy: {test_accuracy:.4f}')

# 2. Predict the class probabilities and class labels
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# %% classification report

print(classification_report(y_pred,y_test,zero_division=0))

# %% plots

# 3. Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="inferno", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 4. Plot the ROC curves for each class
plt.figure(figsize=(10, 8))

for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(to_categorical(y_test, num_classes)[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve for {label} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc='lower right')
plt.show()
