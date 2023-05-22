import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix, classification_report  
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from functions import *


path={
    "bin_train_x":"binary/X_train.csv",
    "bin_train_y":"binary/Y_train.csv",
    "bin_test_x":"binary/X_test.csv",
    #"bin_test_y":"binary/Y_test.csv",
    "multi_train_x":"multi/X_train.csv",
    "multi_train_y":"multi/Y_train.csv",
    "multi_test_x":"multi/X_test.csv",  
    #"multi_test_y":"multi/Y_test.csv"
}
data = {}

for key, value in path.items():
    data[key] = pd.read_csv(value,header=None)

# Accessing the data:
bin_train_x = data['bin_train_x']
bin_train_y = data['bin_train_y']
bin_test_x = data['bin_test_x']
#bin_test_y = data['bin_test_y']
multi_train_x = data['multi_train_x']
multi_train_y = data['multi_train_y']
multi_test_x = data['multi_test_x']
#multi_test_y = data['multi_test_y']





#Output csv results.

col_ranges = [(0,900),(900, 916), (916, 932), (932, 948), (948, 964)]
bin_re_csv=test_Regression_bin(bin_test_x,col_ranges)
multi_re_csv=test_Regression_multi(multi_test_x,col_ranges)
bin_rf_csv=test_RandomForests_bin(bin_test_x,col_ranges)
multi_rf_csv=test_RandomForests_multi(multi_test_x,col_ranges)

# create a list of dataframe names
list_names = ['binary_Regression', 'multi_Regression', 'binary_RandomForsts', 'multi_RandomForsts']

# create a list of dataframes
list_csv = [bin_re_csv, multi_re_csv, bin_rf_csv, multi_rf_csv]

# loop over the list and export each dataframe to a CSV file
for i, data in enumerate(list_csv):
    if isinstance(data, np.ndarray):
        # if the object is a numpy array, convert it to a pandas dataframe
        data = pd.DataFrame(data)
    # export the dataframe to a CSV file
    filename = list_names[i] + '.csv'
    data.to_csv(filename, index=False)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

# Confusion matrix and classification report for the first multi-class problem
cm_m_rf = confusion_matrix(y_m_test_rf, multi_pre_rf)
sns.heatmap(cm_m_rf, annot=True, cmap="Blues", fmt="d", ax=axs[0][0])
axs[0][0].set_title('Multi-RandomForests')
axs[0][0].set(xlabel='Predicted labels', ylabel='True labels')

cm_b_rf= confusion_matrix(y_b_test_rf, bin_pre_rf)
sns.heatmap(cm_b_rf, annot=True, cmap="Blues", fmt="d", ax=axs[0][1])
axs[0][1].set_title('Binary-RandomForests')
axs[0][1].set(xlabel='Predicted labels', ylabel='True labels')

cm_m_re = confusion_matrix(y_m_test_re, multi_pre_re)
sns.heatmap(cm_m_re, annot=True, cmap="Blues", fmt="d", ax=axs[1][0])
axs[1][0].set_title('Multi-Regression')
axs[1][0].set(xlabel='Predicted labels', ylabel='True labels')

cm_b_re = confusion_matrix(y_b_test_re, bin_pre_re)
sns.heatmap(cm_b_re, annot=True, cmap="Blues", fmt="d", ax=axs[1][1])
axs[1][1].set_title('Multi-Regression')
axs[1][1].set(xlabel='Predicted labels', ylabel='True labels')

plt.tight_layout()
plt.show()

