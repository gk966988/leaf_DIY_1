import torch
import numpy as np
a = np.array([[1,1,1],
                 [2,2,2],
                 [3,3,3]])
b = np.array([[4,4,4],
                 [5,5,5],
                  [6,6,6]])
c = []
c.append(a)
c.append(b)
print(c)
# d = torch.cat(c, 1)
d = np.concatenate(c, 1)
print(d.size())
e = d.mean(1)
print()









