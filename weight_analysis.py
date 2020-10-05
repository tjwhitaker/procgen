import matplotlib.pyplot as plt
import numpy as np
import pickle

data = pickle.load(open("4m_logits_weights.pkl", "rb")).cpu().detach().numpy()


for i in range(15):
    print("Mean: ", np.mean(data[i, :]))
