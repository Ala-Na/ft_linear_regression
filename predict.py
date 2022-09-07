import numpy as np
import os
from typing import Tuple
import sys

def predictPrice(theta: np.ndarray, mil: int, vect: bool = False) -> float :
	try:
		if vect == True:
		# Vectorized version
			X = np.insert(np.asarray(float(mil)).reshape(-1, 1), 0, 1.0, axis=1)
			return float(np.dot(X, theta))
		# Non-vectorized
		return theta[0][0] + theta[1][0] * mil
	except:
		print("\033[91m\nOops ! Something went wrong in predict price!\033[0m")
		print("Exiting...")
		exit()

def meanNormalization(mil: int, mean:float, std:float) -> int:
	mil = (mil - mean) / std
	return mil

def getValues() -> Tuple[np.ndarray, float, float] :
	theta = np.asarray([0.0, 0.0])
	mean = 0
	std = 1
	if os.path.isfile('thetas.npz'):
		try:
			values = np.load('thetas.npz')
			theta = values['theta']
			mean = float(values['mean'])
			std = float(values['std'])
		except:
			theta = np.asarray([0.0, 0.0]).reshape(-1, 1)
			mean = 0
			std = 1
		if not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 \
			or theta.shape != (2, 1) :
			theta = np.asarray([0.0, 0.0]).reshape(-1, 1)
			mean = 0
			std = 1
	return theta, mean, std

def checkInput(input: str) -> int:
	try:
		nb = int(input)
		if nb < 0 or nb > 1000000:
			return -1
		return nb
	except ValueError:
		return -1

if __name__ == "__main__" :
	os.system('clear')
	
	argv = sys.argv[1:]
	if len(argv) != 0 and "vect" in argv:
		vect = True
	else:
		vect = False

	print("\n\t\t\033[01;04m~  Welcome to predict programm !  ~\033[0m\n")

	if (vect == False):
		print("\n\033[02mYou're using a non-vectorized implementation\nIf you wish to launch this programm with a vectorized implementation :\npython3 predict.py vect\n\033[0m")
	else :
		print("\n\033[02mYou're using a vectorized implementation\nIf you wish t launch without vectorized implementation, don't set the \"vect\" option\n\033[0m")

	print("\033[93mPlease enter the mileage of car you wish to sold (in kilometers) :\033[0m")
	inp = input("--> ")
	print("You entered {} km".format(inp))
	mil = checkInput(inp)
	if (mil == -1) :
		print("\033[91m\nOops ! That's not a valid positive non-null number or a possible mileage (max 1.000.000 km)!\033[0m")
		print("Exiting...")
		exit()
	theta, mean, std = getValues()
	normalized_mil = meanNormalization(mil, mean, std)
	price = predictPrice(theta, normalized_mil, vect)
	print("\033[92;01m\n--> It's predicted price is of {:.2f} $\n\033[0m".format(price))

