import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# Define the filter 
kernel = torch.tensor(
	[[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]], dtype=torch.float32)
kernel = kernel.reshape(1, 1, 3, 3)

# kernel = torch.tensor(
# 	[[2, 3],
# 	[4, 5]], dtype=torch.float32)
# kernel = kernel.reshape(1, 1, 2, 2)

# Define the bias 
#bias = torch.tensor([5], dtype=torch.float32)

# Define the input image 
image = torch.tensor(
	[[1, 2, 3, 4],
	[5, 6, 7, 8],
	[9, 10, 11, 12],
	[13, 14, 15, 16]], dtype=torch.float32)
image = image.reshape(1, 1, 4, 4)
# image = torch.tensor(
# 	[[1, 2, 3],
# 	[5, 6, 7]], dtype=torch.float32)
# image = image.reshape(1, 1, 2, 3)

# Define the convolution operation 
conv = nn.Conv2d(in_channels=1, out_channels=1, stride=(1,1),kernel_size=(3,3))#, bias=False)

# Set the filter for the convolution operation 
conv.weight = nn.Parameter(kernel)
#conv.bias = nn.Parameter(bias)
# Apply the convolution operation 
output = conv(image) 

# Print the output 
print('Output Shape :',output.shape) 
print('Output \n',output)
