import numpy as np
import pandas as pd
from typing import Tuple
import math
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
		print("\033[91mOops, can't extract datas from {} data file.\033[0m".\
			format(filename))
		print("\033[02;03mHint: Check file rights, file type, ...\033[0m")
		print("Exiting...")
		exit()

def meanNormalization(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
	mean = np.mean(x, axis=0)
	std = np.std(x, axis=0)
	x = (x - mean) / std
	return x, mean, std

def predictedPrices(theta: np.ndarray, x: np.ndarray) -> np.ndarray :
	X = np.insert(x, 0, 1.0, axis=1)
	return np.dot(X, theta)

def getValues() -> Tuple[np.ndarray, float, float] :
	try:
		values = np.load('thetas.npz')
		theta = values['theta']
		mean = float(values['mean'])
		std = float(values['std'])
	except:
		print("\033[91mOops, can't find thetas.npz file or data is corrupted.\033[0m")
		print("Hint: Run train.py")
		print("Exiting...")
		exit()
	if not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 \
		or theta.shape != (2, 1) :
		print("\033[91mOops, can't find thetas.npz file or data is corrupted.\033[0m")
		print("Hint: Run train.py")
		print("Exiting...")
		exit()
	return theta, mean, std

def costCalculation(y: np.ndarray, y_hat: np.ndarray) -> float:
	return np.sum((y_hat - y) * (y_hat - y)) / (2 * y.shape[0])


def mseCalculation(y: np.ndarray, y_hat: np.ndarray) -> float:
	mse = ((y_hat - y) ** 2).mean(axis=None)
	return float(mse)

def rmseCalculation(y: np.ndarray, y_hat: np.ndarray) -> float:
	rmse = math.sqrt(mseCalculation(y, y_hat))
	return float(rmse)

def maeCalculation(y: np.ndarray, y_hat: np.ndarray) -> float:
	mae = (np.absolute(y_hat - y)).mean(axis=None)
	return float(mae)

def r2scoreCalculation(y: np.ndarray, y_hat: np.ndarray) -> float:
	y_bar = y.mean(axis=None)
	r2score = 1 - (np.sum((y_hat - y) ** 2) / np.sum((y - y_bar) ** 2))
	return float(r2score)

if __name__ == "__main__" :
	os.system('clear')
	print("\n\t\t\033[01;04m~  Welcome to precision programm !  ~\033[0m\n")

	print("There's a lot of different calculus to check our model precision")
	print("Let's review some of them !\n")

	(x, y) = getDatas('data.csv')
	if not np.issubdtype(x.dtype, np.number) or x.ndim != 2  \
			or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 \
			or y.shape != (x.shape[0], 1) :
		print("\033[91mOops, there an error in data arrays.\033[0m")
		print("Exiting...")
		exit()
	theta, _, _ = getValues()
	X, _, _ = meanNormalization(x)
	y_hat = predictedPrices(theta, X)

	cont = input('\033[96mPress a key to continue...\033[0m\n')
	os.system('clear')

	print("\n\033[93mCost / loss function:\033[0m\n")
	print("\033[02mIt's the one for which derivative are used used to \nperform \ngradient descent\n\033[0m")
	print("\033[02mPerfect result would be 0 (near impossible except if overfitting)\n\033[0m")
	print("\n\t\t\033[01m1 / 2m * SUM((predictions - targets) ** 2)\033[0m\n")
	print("Result for our model: {}".format(costCalculation(y, y_hat)))

	cont = input('\n\033[96mPress a key to continue...\033[0m\n')
	os.system('clear')

	print("\033[93m\nMean Squared Error function:\033[0m\n")
	print("\033[02mAverage of squared difference\033[0m")
	print("\033[02mIt measures the variance of residuals and penalize large prediction \nerros\n\033[0m")
	print("\033[02mPerfect result would be 0 (near impossible except if overfitting)\n\033[0m")
	print("\n\t\t\033[01m1 / m * SUM((targets - predicted) ** 2)\033[0m\n")
	print("Result for our model: {}".format(mseCalculation(y, y_hat)))

	cont = input('\n\033[96mPress a key to continue...\033[0m\n')
	os.system('clear')

	print("\033[93m\nRoot Mean Squared Error function:\033[0m\n")
	print("\033[02mSquare route of MSE\033[0m")
	print("\033[02mIt measures the standard deviation of residuals and also penalize \nlarge prediction erros\n\033[0m")
	print("\033[02m2nd most used accuracy comparator for linear regression\n\033[0m")
	print("\033[02mPerfect result would be 0 (near impossible except if overfitting)\n\033[0m")
	print("\n\t\t\033[01mSqrt(MSE)\033[0m\n")
	print("Result for our model: {}".format(rmseCalculation(y, y_hat)))

	cont = input('\n\033[96mPress a key to continue...\033[0m\n')
	os.system('clear')

	print("\033[93m\nMean Absolute Error function:\033[0m\n")
	print("\033[02mAverage of the absolute difference\033[0m")
	print("\033[02mIt measures the average of the residuals and doesn't penalize \nlarge prediction erros\n\033[0m")
	print("\033[02mEasier to interpret than MSE/RMSE but non-differentiable, so less \nused\033[0m")
	print("\033[02mPerfect result would be 0 (near impossible except if overfitting)\n\033[0m")
	print("\n\t\t\033[01m1 / m * SUM(abs(targets - predicted))\033[0m\n")
	print("Result for our model: {}".format(maeCalculation(y, y_hat)))

	cont = input('\n\033[96mPress a key to continue...\033[0m\n')
	os.system('clear')

	print("\033[93m\nR2 score (coeeficient of determination or R-squared):\033[0m\n")
	print("\033[02mProportion of variance in the dependent variable\033[0m")
	print("\033[02m(variance = difference between targets and average of predicted \nvalues)\033[0m")
	print("\033[02mCan help to understand the influence of independents variables \non the depends variables\n\033[0m")
	print("\033[02mAn adjusted version of it exists depending on the number of \nindependents variables\033[0m")
	print("\033[02mIt helps to compare two models\033[0m")
	print("\033[02mResult is between 0 and 1 (scale free) with 1 being the perfect \nscore (near impossible except if overfitting)\n\033[0m")
	print("\n\033[01m1 - (SUM((targets - predicted) ** 2) / SUM((targets - mean target value) ** 2))\033[0m\n")
	print("Result for our model: {}".format(r2scoreCalculation(y, y_hat)))

	cont = input('\n\033[95mPress a key to exit !\033[0m\n')
	os.system('clear')
