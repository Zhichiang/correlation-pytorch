# correlation-pytorch-extension
Custom implementation of Corrlation Module WITHOUT compile.

This is a custom implementation of correlation operation. use `corr(...)` to get the result.

correlation operation can be described as:

![](http://latex.codecogs.com/gif.latex?\\corr(A,B)=C)

![](http://latex.codecogs.com/gif.latex?\\c_{bijhw}=\sum_{c}a_{bchw}{\cdot}b_{bch'w'})

![](http://latex.codecogs.com/gif.latex?\\\text{where}{\quad}h'=h+i;{\quad}w'=w+j;{\quad}i,j\in[-k,k])

- b: batch size
- c: channel
- h: height
- w: width
- i: horizontal index of kernel
- j: vertical index of kernel
