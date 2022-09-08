import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

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

def resetMeanNormalization(x: np.ndarray, mean: float, std: float) \
		-> np.ndarray:
	x = (x * std) + mean
	return x

def predictedPrices(theta: np.ndarray, x: np.ndarray, vect: bool = False) \
		-> np.ndarray :
	if vect:
		X = np.insert(x, 0, 1.0, axis=1)
		return np.dot(X, theta)
	res = np.zeros((x.shape[0], 1))
	for ind in range(x.shape[0]):
		value = theta[0][0] + theta[1][0] * x[ind][0]
		res[ind][0] = value
	return res

def gradientValues(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, \
		vect: bool = False) -> np.ndarray:
	if vect:
		X = np.insert(x, 0, 1.0, axis=1)
		return np.dot((np.transpose(X) / x.shape[0]), (y_hat - y))
	theta0 = 0
	theta1 = 0
	for i in range(y.shape[0]):
		theta0 += y_hat[i] - y[i]
		theta1 += (y_hat[i] - y[i]) * x[i]
	theta0 /= x.shape[0]
	theta1 /= x.shape[0]
	return np.asarray([theta0, theta1]).reshape(-1, 1)

def costCalculation(y: np.ndarray, y_hat: np.ndarray, vect:  bool = False) \
		-> float:
	if vect:
		return np.sum((y_hat - y) * (y_hat - y)) / (2 * y.shape[0])
	loss_elem = []
	for yi, yi_hat in zip(y, y_hat):
		loss_elem.append([(yi_hat[0] - yi[0]) ** 2])
	np.asarray(loss_elem)
	cost_value = float(1/(2 * y.shape[0]) * np.sum(loss_elem))
	return cost_value

def fitModel(max_iter: int, alpha: float, x: np.ndarray, y: np.ndarray, \
		vect: bool = False) -> Tuple[np.ndarray, list, list]:
	theta = np.asarray([0, 0]).reshape(-1, 1)
	cost_evolution = []
	theta_history = []
	theta_history.append(theta)
	for i in range(0, max_iter):
		y_hat = predictedPrices(theta, x, vect)
		cost = costCalculation(y, y_hat, vect)
		cost_evolution.append(cost)
		curr_gradient = gradientValues(x, y, y_hat, vect)
		if vect:
			new_theta = theta - alpha * curr_gradient
		else:
			theta0 = theta[0][0] - alpha * curr_gradient[0][0]
			theta1 = theta[1][0] - alpha * curr_gradient[1][0]
			new_theta = np.asarray([theta0, theta1]).reshape(-1, 1)
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

def animate(frame: int, x_cost: list, y_cost: list, x_model: np.ndarray, \
		mean: float, std: float, data_feat: np.ndarray, data_target: np.ndarray, \
		data_feat_norm: np.ndarray, vect: bool = False) -> None:
	if (len(x_cost) == len(cost_evolution) + 1):
		x_cost.clear()
		y_cost.clear()
		ax[0].clear()
	x_cost.append(frame)
	y_cost.append(cost_evolution[frame])
	ax[0].clear()
	ax[0].set_facecolor('black')
	ax[0].plot(x_cost, y_cost, linewidth=4, color='seagreen')
	ax[0].grid()
	ax[0].set_xlim([-1, len(cost_evolution)])
	ax[0].tick_params(color='white', labelcolor='white')
	ax[0].set_ylim([0, max(cost_evolution) * 1.01])
	ax[0].set_xlabel("Epochs / iterations", color='white')
	ax[0].set_ylabel("Cost value", color='white')
	ax[0].set_title("Evolution of cost function while training model", color='white')
	for spine in ax[0].spines.values():
		spine.set_edgecolor('white')

	y_model = predictedPrices(theta_history[frame], x_model, vect)
	loss = predictedPrices(theta_history[frame], data_feat_norm, vect)
	x_model = resetMeanNormalization(x_model, mean, std)
	ax[1].clear()
	ax[1].set_facecolor('black')
	for i in range(loss.shape[0]):
		ax[1].plot([data_feat[i], data_feat[i]], [data_target[i], loss[i]], '--', \
			color='lightcoral', zorder=1, label="element loss")
	ax[1].plot(x_model, y_model, linewidth=3, color='seagreen', zorder=3)
	ax[1].scatter(data_feat, data_target, color='lightgreen', s=70, zorder=2)
	ax[1].grid()
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	ax[1].legend(by_label.values(), by_label.keys())
	ax[1].set_xlim([-10000, 405000])
	ax[1].set_ylim([-500, 10000])
	ax[1].tick_params(color='white', labelcolor='white')
	ax[1].set_xlabel("Mileage (km)", color='white')
	ax[1].set_ylabel("Price", color='white')
	ax[1].set_title("Evolution of model", color='white')
	for spine in ax[1].spines.values():
		spine.set_edgecolor('white')

def drawAnimatedGraphs(cost_evolution: list, theta_history: list, mean: float, \
		std: float, data_feat: np.ndarray, data_target: np.ndarray, \
		data_feat_norm: np.ndarray, vect: bool = False) -> None:
	x_cost = []
	y_cost = []
	x_model = np.linspace(-mean/std, 5.8, 10).reshape(-1, 1)
	ani = animation.FuncAnimation(fig, animate, fargs=(x_cost, y_cost, x_model, \
			mean, std, data_feat, data_target, data_feat_norm, vect), \
				frames=(len(theta_history)), interval=300, repeat=True)
	plt.show()


if __name__ == "__main__" :
	os.system('clear')

	argv = sys.argv[1:]
	if len(argv) != 0 and "vect" in argv:
		vect = True
	else:
		vect = False

	print("\n\t\t\033[01;04m~  Welcome to training programm !  ~\033[0m\n")

	if (vect == False):
		print("\n\033[02mYou're using a non-vectorized implementation\nIf you wish to launch this programm with a vectorized implementation :\npython3 train.py vect\n\033[0m")
	else :
		print("\n\033[02mYou're using a vectorized implementation\nIf you wish to launch without vectorized implementation, don't set the \"vect\" option\n\033[0m")


	print("Here, we will perform a single variable linear regression model.")

	cont = input('\n\033[96mPress a key to continue...\033[0m\n')
	os.system('clear')

	print("\033[93m\n--> How does it work ?\033[0m")

	print("\nIt's an equation under the format :\n\n\t\t\033[01my_hat = theta0 + theta1 * x\033[0m\n")

	print("Here, we choose random parameters theta0 and theta1, and we want to \naffine them")
	print("In other words: Choose parameters which give better prediction \n(y_hat), closer to expected result")
	print("For that, we calculate the cost of the model for given parameters \nthetas (= global difference between predicted and expected results)")
	print("Then, we affinate those parameters thanks to the derivative \nproperties of the cost function\n")

	print("\033[93m\n--> Let's perform it !\n\033[0m")

	cont = input('\n\033[96mPress a key to continue...\033[0m\n')
	os.system('clear')

	print("\033[93m\n1- First let's get the data and store it into arrays\033[0m")
	print("\033[02mAn array of features (mileage) and an array of targets (prices)\033[0m")
	print("\033[02mBecause we only have one variable here (mileage), features array \nis of shape 1 * number of examples\033[0m")
	print("\033[02mIf we had n variables or features, it would be an array of shape \nm * number of examples\033[0m")
	print("\033[02mTarget array must always be of shape m * 1 (vector)\033[0m")
	(x, y) = getDatas('data.csv')

	if not np.issubdtype(x.dtype, np.number) or x.ndim != 2  \
			or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 \
			or y.shape != (x.shape[0], 1) :
		print("\033[91mOops, there an error in data arrays.\033[0m")
		print("Exiting...")
		exit()

	print("\n\033[04mArray x (features) :\033[0m\n\n{}".format(x.reshape(1, -1)))
	print("\n\033[04mArray y (targets) :\033[0m\n\n{}".format(y.reshape(1, -1)))

	cont = input('\n\033[96mPress a key to continue...\033[0m\n')
	os.system('clear')

	print("\033[93m\n2 - As big values can screw up computing calculations, let's scale \nour features array\033[0m")
	print("\033[02mHere, we use mean normalization (substract mean and divide by standard \nvariation)\033[0m")
	print("\033[02mIt works well with normally distributed data and brings most values in \n-0.5 - 0.5 range\033[0m")

	X, mean, std = meanNormalization(x)
	print("\n\033[04mScaled array x :\033[0m\n\n{}".format(X.reshape(1, -1)))

	cont = input('\n\033[96mPress a key to continue...\033[0m\n')
	os.system('clear')

	print("\033[95m\n3 - !NOTE! If we had more datas and that feature engineering was \npossible, we should cut our data set in training and testing set\033[0m")
	print("\033[02m- Training set is used to train differents models and affinate our \nmodels parameters\033[0m")
	print("\033[02mIf we used ALL our data set for training, we could end up with a \nproblem called 'overfitting'\033[0m")
	print("\033[02mIt's when our model is perfect for the data set but it can't be \ngeneralized to other data\033[0m")
	print("\033[02m- Testing set is kept untouched and will be used to calculate our \nbest model accuracy after training\033[0m")
	print("\033[02m\nExample of proportions choosen from initial data set : \n80% training set, 20% testing set\033[0m")

	cont = input('\n\033[96mPress a key to continue...\033[0m\n')
	os.system('clear')

	print("\033[95m\n4 - !NOTE! If it wasn't for predict.py, we could perform here some \nfeature engineering\033[0m")
	print("\033[02mOur perfect model may not be a straight curve\033[0m")
	print("\033[02mIn order to achieve non-straight curve, we could train different \npolynomial forms of our features array, and adding them to our \ncurrent array\033[0m")
	print("\033[02mThe best one, with less difference will be choosen later\033[0m")
	print("\033[02mFeature engineering is a vast subject and many technics other than \npolynomial form exist\033[0m")
	print("\033[02mFor example, if we had more than one variable we could use crossed \nform (feature 1 * feature 2), ...\033[0m")
	print("\033[02mHere, we can't use it as it won't work with the prediction program \nanymore (polynmial degree should be shared)\033[0m")

	cont = input('\n\033[96mPress a key to continue...\033[0m\n')
	os.system('clear')

	print("\033[93m\n5 - Now, let's train our model !\033[0m")
	print("\033[02mFor the magic to happen, we need to update theta0 and theta1 on a \npreviously defined max number of iterations\033[0m")
	print("\033[02mWe can actually stop the iterations when thetas stop evolving between \ntwo iterations because it means it's their best values/the cost function \nvalue is at it's lowest\033[0m")
	print("\033[02mFor each iteration, we calculate new thetas thanks to gradient descent\033[0m")
	print("\033[02mWe obtained gradient values thanks to derivatives properties of cost function\033[0m")
	print("\033[02mThose gradient values give the \"direction\" in which thetas should evolve to \nreduce cost \033[0m")
	print("\033[02mThen, we update simultaneously every theta in the corresponding direction \nby substracting this direction from original thetas\033[0m")
	print("\033[02mThe importance of the update is set by a learning rate alpha which is \ncommonly a small number under 1\033[0m")
	print("\033[02mAlpha is important as a big alpha can miss the optimal thetas values and \na small alpha can make convergence happen slower\033[0m")
	print("\033[02mHere, thetas are both 0 at beginning, number of max iterations is 50000 and \nlearning rate is 0.5\033[0m")

	print("\nEquations details for each step:\n\n\033[01m   theta0 = theta0 - alpha * 1/m SUM(prediction[i] - target[i])\n   theta1 = theta1 - alpha * 1/m SUM((prediction[i] - target[i]) * feature[i])\033[0m\n")


	theta, cost_evolution, theta_history = fitModel(25000, 0.5, X, y, vect)
	print("\n\033[94mWe obtained new thetas\033[0m:\n{}\n".format(theta.reshape(1, -1)))
	saveValues(theta, mean, std)

	cont = input('\n\033[96mPress a key to continue...\033[0m\n')
	os.system('clear')

	print("\033[93m\n5 - Let's trace the cost evolution and model evolution !\033[0m")
	fig, ax = plt.subplots(1, 2, figsize=(30,10))
	fig.set_facecolor('black')
	drawAnimatedGraphs(cost_evolution, theta_history, mean, std, x, y, X, vect)

	cont = input('\n\033[96mPress a key to continue...\033[0m\n')
	os.system('clear')

	print("\n\033[92m--> You saw everything !\nNow you should be ready to do your own linear regression !\033[0m")

	cont = input('\n\033[95mPress a key to exit\033[0m\n')
	os.system('clear')
