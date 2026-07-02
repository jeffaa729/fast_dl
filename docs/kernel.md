General : a kernel function is defined, then a launch_kernel function to call with configuration of grid and block size.

1. Choice of block and grid size:

2. Activations : 
- ReLU
naive : each thread get 1 element

- LeakyReLU
naive : each thread get 1 element

- ReLU backward
naive : each thread get 1 element, if x > 0 then get grad else 0

- Sigmoid 
naive : each thread get 1 element

- Tanh
naive : each thread get 1 element

- GeLU
naive : each thread get 1 element

3. Bias
add_row_bias :
this is a part of the linear 
for adding a constant bias, a bias shape [col] has to add to every row.
`idx % cols` to get which col the idx in.

backward : 
forward formula = output = input + bias
```
output[0,0] = input[0,0] + bias[0]
output[1,0] = input[1,0] + bias[0]
```
bias[0] affects multiple outputs so during backward, gradient for bias[0] must collect both:
```
grad_bias[0] = grad_output[0,0] + grad_output[1,0]
```
so for all columns :
```
grad_bias[0] = sum all rows in column 0
grad_bias[1] = sum all rows in column 1
grad_bias[2] = sum all rows in column 2
grad_input = grad_ouput
```
```
{   
    grad_output, // grad of input 
    grad_bias // grad of bias
}
```
