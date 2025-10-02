import zipfile
import os

zip_path = 'archive.zip'  
extract_path = 'archive'         


os.makedirs(extract_path, exist_ok=True)


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("archive extracted successfully!")
import os
for root, dirs, files in os.walk('archive'):
    for file in files:
        print(os.path.join(root, file))
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(true_classes, predicted_classes, target_names=class_labels))
precision    recall  f1-score   support

      glioma       0.94      0.92      0.93       300
  meningioma       0.91      0.85      0.88       306
     notumor       0.95      1.00      0.97       405
   pituitary       0.98      0.99      0.98       300

    accuracy                           0.94      1311
   macro avg       0.94      0.94      0.94      1311
weighted avg       0.94      0.94      0.94      1311
