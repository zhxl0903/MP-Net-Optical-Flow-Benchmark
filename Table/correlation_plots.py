import numpy as np
import os
import matplotlib.pyplot as plt

flow_names = ['EpicFlow', 'FlowNet2', 'LDOF', 'PWC-Net', 'SelFlow', 'SPyNet']
sintel_final_test_aee = [6.29, 6.02, None, 5.13, 4.26, 8.36]
J = [0.677, 0.698, 0.699, 0.677, 0.701, 0.646]
F = [0.643, 0.669, 0.665, 0.649, 0.674, 0.602]
T = [0.739, 0.668, 0.684, 0.773, 0.724, 0.703]

fig, axs = plt.subplots(3)

# Plots AEE vs. Region Similarity (J), Contour Accuracy (F), and Temporal Stability (T)
axs[0].scatter(sintel_final_test_aee, J)
axs[0].set_ylabel('Region Similarity (J)', size=8)
axs[1].scatter(sintel_final_test_aee, F)
axs[1].set_ylabel('Contour Accuracy (F)', size=8)
axs[2].scatter(sintel_final_test_aee, T)
axs[2].set_xlabel(xlabel='AEE on Sintel Final Test', size=8)
axs[2].set_ylabel(ylabel='Temporal Stability (T)', size=8)

fig.savefig('./correlation.png', bbox_inches='tight')
plt.tight_layout()
plt.show()
