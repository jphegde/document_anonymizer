import numpy as np
import matplotlib.pyplot as plt

x_axis = ['linear', 'poly', 'rbf', 'sigmoid']
accuracy = [80.71, 72.86, 72.86, 72.86]
precision = [] 
#plt.plot(x_axis, sse, marker='o')
plt.title('Accuracy and precision vs kernel of SVM')
plt.plot(x_axis, accuracy,'g--', label="Accuracy")
plt.plot(x_axis, precision,'r-o', label="Precision")
plt.legend()
plt.xlabel('kernel')
#plt.ylabel('sum of squared distance from cluster center')
plt.ylabel('Accuracy and Precision in %')
plt.show()