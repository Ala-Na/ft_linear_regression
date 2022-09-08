# ft_linear_regression

A 42 school project, from the machine learning / artificial intelligence branch.
![GIFLinearReg](https://user-images.githubusercontent.com/67599180/189076214-3aeaf657-429e-42ad-9197-c89b6887b190.gif)
Project screen capture 

## Purpose
A two-day project to carry out a univariate linear regression on a dataset containing cars (data.csv). The characteristic employed is the number of kilometres and the target is the price.
The aim is to construct an algorithm to perform a linear regression and forecast car prices based on a given mileage.


## Programs

### :brain: predict.py
```python3 predict.py```

```python3 predict.py vect```

A simple program which take an input (number of mileage) and returns a predicted price.
Default values for weights are set to 0. If program was previously trained, calculated weight will be chosen.
With ```vect``` argument, it will perform vectorized calculations. 

### :mechanical_arm: train.py
```python3 train.py```

```python3 train.py vect```

A program to perform training of univariate linear regression model.
A step-by-step guide will explains why and how things are done for linear regression. Additionnal step, not done in this program, are also explained.
With ```vect``` argument, it will perform vectorized calculations. 

### :100: accuracy.py
```python3 accuracy.py```

A program which explains the differents accuracy measures available and perform it for our current data set.



## Language used
Python :snake:
Why ? Because it's the main language used in data science and machine learning nowadays.


## Libraries used
- Numpy
- Pandas
- matplotlib
