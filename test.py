import cv2
import torch
import matplotlib.pyplot as plt

path = input("path:")
color = cv2.imread(path)
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
gray =  cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
torch_color = torch.from_numpy(color)
torch_gray = torch.from_numpy(gray)

print(type(color))
print(color.size)
print(color.shape)
plt.imshow(color)
plt.show()

print(type(gray))
print(gray.size)
print(gray.shape)
plt.imshow(gray)
plt.gray()
plt.show()

print(type(torch_color))
print(torch_color.size)
print(torch_color.shape)

print(type(torch_gray))
print(torch_gray.size)
print(torch_gray.shape)
