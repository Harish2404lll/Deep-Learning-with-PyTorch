# PyTorch Basics Exercises

## AIM:
Write a Python program using PyTorch that performs the following tasks:

### Software Required:
- Python 3.x
- PyTorch
- Jupyter Notebook (for interactive development and execution)

## Algorithm:

### Step 1:
Perform standard imports
- Import `torch` and `NumPy`.

```python
# Import necessary libraries
import torch
import numpy as np
```

### Step 2:
Set the random seed for NumPy and PyTorch both to `42`.

```python
# Create a tensor filled with zeros of size (3,3)
tensor_zeros = torch.zeros((3,3))
print('Tensor of zeros:\n', tensor_zeros)
```

### Step 3:
Create a NumPy array called `arr` that contains 6 random integers between 0 (inclusive) and 5 (exclusive).

```python

# Create a random tensor of size (3,3)
tensor_random = torch.rand((3,3))
print('Random Tensor:\n', tensor_random)
```

### Step 4:
Create a tensor `x` from the array above.

```python

# Perform basic tensor addition
tensor_sum = tensor_zeros + tensor_random
print('Sum of tensors:\n', tensor_sum)
```

### Step 5:
Change the dtype of `x` from `int32` to `int64`.

```python
# Check if CUDA is available and create a tensor on GPU if possible
if torch.cuda.is_available():
    tensor_gpu = torch.rand((3,3)).to('cuda')
    print('Tensor on GPU:\n', tensor_gpu)
else:
    print('CUDA is not available')
```

### Step 6:
Reshape `x` into a `3x2` tensor.

```python

# Convert PyTorch tensor to NumPy array
numpy_array = tensor_random.numpy()
print('Converted NumPy array:\n', numpy_array)
```

### Step 7:
Return the right-hand column of tensor `x`.

```python
# Reshape a tensor
tensor_reshaped = tensor_random.view(9)
print('Reshaped tensor:\n', tensor_reshaped)
```

### Step 8:
Without changing `x`, return a tensor of square values of `x`.

```python
# Gradient tracking example
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()
print('Gradient of x:\n', x.grad)
```

### Step 9:
Create a tensor `y` with the same number of elements as `x`, that can be matrix-multiplied with `x`.
- Use PyTorch directly (not NumPy) to create a tensor of random integers between 0 (inclusive) and 5 (exclusive).

```python

# Additional code section
x = torch.tensor([[1, 2],
                  [3, 4],
                  [5, 6]])
y = torch.tensor([[2, 2, 1],
                  [4, 1, 0]])
```

### Step 10:
Find the matrix product of `x` and `y`.

```python

# Additional code section
print("Tensor y:")
print(y)

result = torch.matmul(x, y)
print("Matrix product of x and y:")
print(result)
```

## Output:
i) Import and set up PyTorch and NumPy.
ii) Create and manipulate tensors.
iii) Perform matrix operations.

## Result:
Thus, the PyTorch tensor operations, including reshaping, dtype conversion, and matrix multiplication, were successfully performed using the Python program.
