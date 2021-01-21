#!/usr/bin/env python
# coding: utf-8

import numpy as np


test = [
    [[1,2,3], [4,5,6], [7,8,9]],
    [[9,8,7], [6,5,4], [3,2,1]],
]

test_b = [
    [[1,2,3], [4,5,6], [7,8,9]],
    [[9,8,7], [6,5,4], [3,2,1]],
]


test


test_np = np.array(test)
test_nb = test_np.copy()

print(test_np)
print(test_nb)


test_np.shape


test_np[1,:,:]


test_np = np.append(test_np, "test")



print(test_np)
print(test_nb)


if (test_np == test_nb):
    print("1")
else:
    print("2")


test_np.tolist() == test_nb.tolist()


(test_np.reshape(18))


len(test_np)




