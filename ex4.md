使用反向传播的前馈神经网络，自动学习神经网络的参数。

### 神经网络

- 向前传播

<img src="C:\Users\chall\AppData\Roaming\Typora\typora-user-images\image-20211103150430175.png" alt="image-20211103150430175" style="zoom:90%;float:left;" />

- 实现神经网络的代价函数和梯度函数

```python
#向前传播函数
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    
    a1 = np.insert(X, 0, values=np.ones(m), axis=1) # 插入一列1元素，偏置
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    
    return a1, z2, a2, z3, h
```

- 代价函数

<img src="C:\Users\chall\AppData\Roaming\Typora\typora-user-images\image-20211103150855809.png" alt="image-20211103150855809" style="zoom:100%;float:left;" />

```python
def cost(theta1, theta2 , input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    
    J = J / m
    
    return J
```

- 正则化代价函数

<img src="C:\Users\chall\AppData\Roaming\Typora\typora-user-images\image-20211103151240544.png" alt="image-20211103151234224" style="zoom:100%;float:left;" />

```python
def costReg(theta1, theta2, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # 向前传播
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    
    J = J / m
    
    # 加正则项
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    
    return J
```

### 反向传播

实现反向传播的算法，来计算神经网络代价函数的梯度。获得了梯度的数据，我们就可以使用工具库来计算代价函数的最小值。

- sigmoid梯度

<img src="C:\Users\chall\AppData\Roaming\Typora\typora-user-images\image-20211103151749932.png" alt="image-20211103151749932" style="zoom:120%;float:left" />

- 正则化神经网络

```python
def backpropReg(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    
    J = J / m
    
    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    
    # perform backpropagation
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)
        
        d3t = ht - yt  # (1, 10)
        
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)
        
        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
        
    delta1 = delta1 / m
    delta2 = delta2 / m
    
    # add the gradient regularization term
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m
    
    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    
    return J, grad
```

```python
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

for t in range(m):
	a1t = a1[t, :]  # (1, 401)
    z2t = z2[t, :]  # (1, 25)
    a2t = a2[t, :]  # (1, 26)
    ht = h[t, :]  # (1, 10)
    yt = y[t, :]  # (1, 10)

    d3t = ht - yt  # (1, 10)

    z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
    d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

    delta1 = delta1 + (d2t[:, 1:]).T * a1t
    delta2 = delta2 + d3t.T * a2t
```

