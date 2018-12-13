import numpy as np
import matplotlib.pyplot as plt
import csv

train_ratio = 90.0

f = open(str(train_ratio) + '_history.txt')
X = []

reader = csv.reader(f, delimiter=',')
for row in reader:
    X.append(np.array(row))

X = np.array(X).astype('float')

x = range(0, X.shape[0])

plt.figure(figsize=(12., 9.))

plt.plot(x, X[:, 0], c='r', linewidth=2.0)
plt.plot(x, X[:, 2], c='g', linewidth=2.0)
# plt.axis('equal')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend(['validation set', 'train set'])
plt.title('loss versus iteration')
plt.savefig(str(train_ratio) + '_loss.png', dpi=100)
plt.close()

plt.figure(figsize=(12., 9.))

plt.plot(x, X[:, 1], c='r', linewidth=2.0)
plt.plot(x, X[:, 3], c='g', linewidth=2.0)
# plt.axis('equal')
plt.xlabel('iteration')
plt.ylabel('acc')
plt.title('accuracy versus iteration')
plt.legend(['validation set', 'train set'])
plt.savefig(str(train_ratio) + '_acc.png', dpi=100)
plt.close()
