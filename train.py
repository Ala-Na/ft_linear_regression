import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def getDatas(filename: str) -> Tuple[np.ndarray, np.ndarray]:
	if not os.path.isfile(filename):
		print("\033[91mOops, can't find {} data file.\033[0m".format(filename))
		print("Exiting...")
		exit()
	try:
		data_df = pd.read_csv(filename)
		data_array = np.asarray(data_df)
		return (data_array[:, 0].reshape(-1, 1), data_array[:, 1].reshape(-1, 1))
	except:
		print("\033[91mOops, can't extract datas from {} data file.\033[0m".format(filename))
		print("\033[02;03mHint: Check file rights, file type, ...\033[0m")
		print("Exiting...")
		exit()

def meanNormalization(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
	mean = np.mean(x, axis=0)
	std = np.std(x, axis=0)
	x = (x - mean) / std
	return x, mean, std

def resetMeanNormalization(x: np.ndarray, mean: float, std: float) -> np.ndarray:
	x = (x * std) + mean
	return x

def predictedPrices(theta: np.ndarray, x: np.ndarray) -> np.ndarray :
	X = np.insert(x, 0, 1.0, axis=1)
	return np.dot(X, theta)

def gradientValues(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
	X = np.insert(x, 0, 1.0, axis=1)
	return np.dot((np.transpose(X) / x.shape[0]), (y_hat - y))

def costCalculation(y: np.ndarray, y_hat: np.ndarray) -> float:
	return np.sum((y_hat - y) * (y_hat - y)) / (2 * y.shape[0])

def fitModel(max_iter: int, alpha: float, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, list, list]:
	theta = np.asarray([0, 0]).reshape(-1, 1)
	cost_evolution = []
	theta_history = []
	theta_history.append(theta)
	for i in range(0, max_iter):
		y_hat = predictedPrices(theta, x)
		cost = costCalculation(y, y_hat)
		cost_evolution.append(cost)
		curr_gradient = gradientValues(x, y, y_hat)
		new_theta = theta - alpha * curr_gradient
		if ((new_theta == theta).all()):
				print("\033[92m\nConvergence reached !\033[0m")
				break
		theta = new_theta
		theta_history.append(theta)
	return theta, cost_evolution, theta_history

def saveValues(theta: np.ndarray, mean: float, std:float) -> None:
	if (os.path.isfile('thetas.npz')):
			os.remove('thetas.npz')
	try:
		np.savez('thetas.npz', theta=theta, mean=mean, std=std)
	except:
		print("\033[91mOops, can't save thetas into thetas.npz.\033[0m")
		exit()

def animateCost(frame: int, x: list, y: list) -> None:
	x.append(frame)
	y.append(cost_evolution[frame])
	ax.clear()
	ax.set_facecolor('black')
	ax.plot(x, y, linewidth=4, color='seagreen')
	ax.grid()
	ax.set_xlim([-1, len(cost_evolution)])
	ax.set_ylim([0, max(cost_evolution) * 1.01])
	ax.set_xlabel("Epochs / iterations")
	ax.set_ylabel("Cost value")
	ax.set_title("Evolution of cost function while training model")

def drawAnimatedCostEvolution(cost_evolution: list) -> None:
	x = []
	y = []
	ani = animation.FuncAnimation(fig, animateCost, fargs=(x, y,), frames=(len(cost_evolution)), \
		interval=100, repeat=False)
	plt.show()

def animateModel(frame: int, x: list, mean: float, std: float) -> None:
	print(frame)
	y = predictedPrices(theta_history[frame], x)
	print(theta_history[frame])
	x = resetMeanNormalization(x, mean, std)
	ax.clear()
	ax.set_facecolor('black')
	ax.plot(x, y, linewidth=1, color='seagreen')
	ax.grid()
	ax.set_xlim([-1000, 80000])
	ax.set_ylim([-1000, 10000])
	ax.set_xlabel("Mileage (km)")
	ax.set_ylabel("Price")
	ax.set_title("Evolution of model")

def drawAnimatedModel(theta_history: list, mean: float, std: float) -> None:
	print(theta_history)
	x = np.linspace(-mean/std, 6, 10).reshape(-1, 1)
	ani = animation.FuncAnimation(fig, animateModel, fargs=(x, mean, std,), frames=(len(theta_history)), \
		interval=1000, repeat=False)
	plt.show()

if __name__ == "__main__" :
	print("\n\t\t\033[01;04m~  Welcome to training programm !  ~\033[0m\n")

	print("Here, we will perform a single variable linear regression model.")
	print("--> How does it work ?")

	print("\nIt's an equation under the format :\n\n\t\t\033[01my_hat = theta0 + theta1 * x\033[0m\n")

	print("Here, we choose random parameters theta0 and theta1, and we want to \naffine them.")
	print("In other words: Choose parameters which give better prediction \n(y_hat), closer to expected result")
	print("For that, we calculate the cost of the model for given parameters \nthetas == global difference between predicted and expected results")
	print("And then, we affinate those parameters thanks to the derivative \nproperties of the cost function.\n")

	print("\n--> Let's perfom it !\n")

	print("\033[93m\n1- First let's recuperate the data and store it into arrays\033[0m")
	print("\033[02mAn array of features (mileage) and an array of targets (prices)\033[0m")
	print("\033[02mBecause we only have one variable here (mileage), features array \nis of shape 1 * number of examples\033[0m")
	print("\033[02mIf we had n variables or features, it would be an array of shape \nm * number of examples\033[0m")
	print("\033[02mTarget array must always be of shape m * 1 (vector)\033[0m")
	(x, y) = getDatas('data.csv')

	if not np.issubdtype(x.dtype, np.number) or x.ndim != 2  \
		or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape != (x.shape[0], 1) :
		print("\033[91mOops, there an error in data arrays.\033[0m")
		print("Exiting...")
		exit()

	print("\nArray x (features) :\n{}".format(x.reshape(1, -1)))
	print("\nArray y (targets) :\n{}".format(y.reshape(1, -1)))


	print("\033[93m\n2 - As big values can screw up computing calculations, let's scale \nour features array\033[0m")
	print("\033[02mHere, we use mean normalization (substract mean and divide by standard \nvariation)\033[0m")
	print("\033[02mIt works well with normally distributed data and brings most values in \n-0.5 - 0.5 range\033[0m")

	X, mean, std = meanNormalization(x)
	print("\nScaled array x :\n{}".format(X.reshape(1, -1)))

	print("\033[95m\n3 - !NOTE! If we had more datas and that feature engineering was \npossible, we should cut our data set in training and testing set\033[0m")
	print("\033[02m- Training set is used to train differents models and affinate our \nmodels parameters\033[0m")
	print("\033[02mIf we used ALL our data set for training, we could end up with a \nproblem called 'overfitting'\033[0m")
	print("\033[02mIt's when our model is perfect for the data set but it can't be \ngeneralized to other data\033[0m")
	print("\033[02m- Testing set is kept untouched and will be used to calculate our \nbest model accuracy after training\033[0m")
	print("\033[02m\nProportions choosen from initial data set : 80% training set, 20% \ntesting set\033[0m")


	print("\033[95m\n4 - !NOTE! If it wasn't for predict.py, we could perform here some \nfeature engineering\033[0m")
	print("\033[02mOur perfect model may not be a straight curve\033[0m")
	print("\033[02mIn order to achieve non-straight curve, we could train different \npolynomial forms of our features array, and adding them to our \ncurrent array\033[0m")
	print("\033[02mThe best one, with less difference will be choosen later\033[0m")
	print("\033[02mFeature engineering is a vast subject and many technics other than \npolynomial form exist\033[0m")
	print("\033[02mFor example, if we had more than one variable we could use crossed \nform (feature 1 * feature 2), ...\033[0m")
	print("\033[02mHere, we can't use it as it won't work with the prediction program \nanymore (polynmial degree should be shared)\033[0m")

	print("\033[93m\n5 - Now, let's train our model !\033[0m")
	print("\033[02mFor the magic to happen, we need to update theta0 and theta1 on a \nprevisously defined max number of iterations\033[0m")
	print("\033[02mWe can actually stop the iterations when thetas stop evolving between \ntwo iterations because it means it's their best values/the cost function\n value is at it's lowest\033[0m")
	print("\033[02mFor each iteration, we calculate new thetas thanks to gradient descent\033[0m")
	print("\033[02mWe obtained gradient values thanks to derivative properties of cost function\033[0m")
	print("\033[02mThose gradient values give the \"direction\" in which thetas should evolve to \nreduce cost \033[0m")
	print("\033[02mThen, we update simultaneously every theta in the corresponding direction \nby substracting this direction from original thetas.\033[0m")
	print("\033[02mThe importance of the update is set by a learning rate alpha which is \ncommonly a small number under 1\033[0m")
	print("\033[02mAlpha is important as a big alpha can miss the optimal thetas value and \na small alpha can make convergence happen slower\033[0m")
	print("\033[02mHere, beginning thetas are both 0, number of max iterations is 50000 and \nlearning rate is 0.5\033[0m")

	theta, cost_evolution, theta_history = fitModel(25000, 0.5, X, y)
	print("\n\033[94mWe obtained new thetas\033[0m:\n{}".format(theta.reshape(1, -1)))
	saveValues(theta, mean, std)

	print("\033[93m\n5 - Let's trace the cost evolution\033[0m")
	fig, ax = plt.subplots(figsize=(15,10))
	fig.set_facecolor('silver')
	drawAnimatedCostEvolution(cost_evolution)

	print("\033[93m\n6 - Let's trace the data set and model\033[0m")
	fig, ax = plt.subplots(figsize=(15,10))
	fig.set_facecolor('silver')
	drawAnimatedModel(theta_history, mean, std)
