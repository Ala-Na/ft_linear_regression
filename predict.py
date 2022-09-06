import numpy as np
import os

def predictPrice(theta: np.ndarray, mil: int) -> float :
	try:
		X = np.insert(float(mil), 0, 1.0).reshape(1, -1)
		return float(np.dot(X, theta))
	except:
		print("\033[91m\nOops ! Something went wrong !\033[0m")
		print("Exiting...")
		exit()

def getThetas() -> np.ndarray :
	theta = np.asarray([0.0, 0.0])
	if os.path.isfile('thetas.npz'):
		saved_theta = np.load('thetas.npz')
		theta = saved_theta['theta']
		if not np.isubdtype(theta.dtype, np.numer) or theta.ndim != 2 \
			or theta.shape != [2, 1]:
			theta = np.asarray([0.0, 0.0])
	return theta

def checkInput(input: str) -> int:
	try:
		nb = int(input)
		if nb <= 0 or nb > 1000000:
			return -1
		return nb
	except ValueError:
		return -1

if __name__ == "__main__" :
	print("\n\t\t\033[01;04m~  Welcome to predict programm !  ~\033[0m\n")
	print("\033[93mPlease enter the mileage of car you wish to sold (in kilometers) :\033[0m")
	inp = input("--> ")
	print("You entered {} km".format(inp))
	mil = checkInput(inp)
	if (mil == -1) :
		print("\033[91m\nOops ! That's not a valid positive non-null number or a possible mileage (max 1.000.000 km)!\033[0m")
		print("Exiting...")
		exit()
	theta = getThetas()
	price = predictPrice(theta, mil)
	print("\033[92;01m\n--> It's predicted price is of {} $\n\033[0m".format(price))

