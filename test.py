import pyflow
import numpy as np

a = np.zeros((64, 64, 3), np.float32)
b = np.ones((64, 64, 3), np.float32)

alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
# 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
colType = 0

u, v, im2W = pyflow.coarse2fine_flow(
    a, b, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType)

flow = np.concatenate((u[..., None], v[..., None]), axis=2)
np.save('outFlow.npy', flow)
