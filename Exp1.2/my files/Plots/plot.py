import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

total = pd.read_csv("./emnist_10x10_original_1patch.txt", delimiter = ' ').values
train_acc = total[:45,0]
val_acc = total[:45,1]

total2 = pd.read_csv("./emnist_fisheye_2patches_10x10.txt", delimiter = ' ').values
train_acc2 = total2[:45,0]
val_acc2= total2[:45,1]

epoch = np.linspace(1,45,45, dtype=int)


plt.show()
plt.figure()
plt.subplot(1,2,1)
plt.plot(epoch,train_acc, label="original with 10x10 single patch")
plt.plot(epoch,train_acc2, label="fisheye with 10x10 single patch" )
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc=4)
plt.title('training accuracy comparison')
plt.subplot(1,2,2)
plt.plot(epoch,val_acc,label="original with 10x10 single patch")
plt.plot(epoch,val_acc2, label="fisheye with 10x10 single patch")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc=4)
plt.title('Validation accuracy comparison')
plt.show()