from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from facerecognition import load_metadata
import matplotlib.pyplot as plt

IMAGE_DIR = './images/'
metadata = load_metadata(IMAGE_DIR)
distances = [] # squared L2 distance between pairs
identical = [] # 1 if same identity, 0 otherwise
embedded = np.load('emmbedded.npy')

def distance(emb1, emb2):
        return np.sum(np.square(emb1 - emb2))


num = len(metadata)

for i in range(num - 1):
	for j in range(1, num):
		distances.append(distance(embedded[i], embedded[j]))
		identical.append(1 if metadata[i].name == metadata[j].name else 0)

distances = np.array(distances)
identical = np.array(identical)

thresholds = np.arange(0.3, 1.0, 0.01)

f1_scores = [f1_score(identical, distances < t) for t in thresholds]
acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

opt_idx = np.argmax(f1_scores) 
opt_tau = thresholds[opt_idx] 
opt_acc = accuracy_score(identical, distances < opt_tau)
print(opt_tau)
# plt.figure(figsize=(8,3))
plt.plot(thresholds, f1_scores, label='F1 score')
plt.plot(thresholds, acc_scores, label='Accuracy')
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}')
plt.xlabel('Distance threshold')

plt.legend()
plt.show()