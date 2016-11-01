# Kaggle Digit Recognizer
kaggle competition - Digit Recognizer(https://www.kaggle.com/c/digit-recognizer)
# Model
1. extend training datas
<p>Using translation operation and reshape the training/testing data(28x28) to 24x24, and get four time training/testing datas.</p>

2. tensorflow deep convolutional neural network.
<p>input + conv/conv/pool/relu/dropout/ + conv/conv/pool/relu/dropout/ + fc/fc/fc/dropout/ readout</p>

- input: 24 x 24 x 1
- conv1_1    : 32 kernels of size = 3x3x1, strides = 1, padding = 'SAME'
- conv1_2    : 32 kernels of size = 3x3x32, strides = 1, padding = 'SAME'
- max-pooling: neighborhoods = 2 x 2, strides = 1
- conv2_1    : 64 kernels of size = 3x3x32, strides = 1, padding = 'SAME'
- conv2_2    : 64 kernels of size = 3x3x64, strides = 1, padding = 'SAME'
- max-pooling: neighborhoods = 2 x 2, strides = 1
- fc1        : 6x6x64 x 256
- fc2        : 256 x 1024
- fc3        : 1024 x 256
- readout    : 256 x 10

# Result
Finally we get 0.99271 test accuracy.
![result](https://github.com/SunnyMarkLiu/DigitRecognizer/blob/master/tf/advance/result.png)

# Moreover
How to get 1.0 test accuracy?
![result](https://github.com/SunnyMarkLiu/DigitRecognizer/blob/master/tf/advance/result1.png)
