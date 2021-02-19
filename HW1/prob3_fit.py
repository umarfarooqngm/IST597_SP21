import os  
import tensorflow as tf 
import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np
'''
IST 597: Foundations of Deep Learning
Problem 1: Univariate Regression


    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# NOTE: You can work in pure eager mode or Keras or even hybrid
# NOTE : Use tf.Variable when using gradientTape, if you build your own gradientTape then use simple linear algebra using numpy or tensorflow math

# meta-parameters for program
#Change variables to tf.constant or tf.Variable whenever needed
trial_name = 'p6_reg0' # will add a unique sub-string to output of this program
degree = 6 # p, degree of model (LEAVE THIS FIXED TO p = 6 FOR THIS PROBLEM)
beta = 500 # regularization coefficient
alpha = 0.001 # step size coefficient
n_epoch = 5000 # number of epochs (full passes through the dataset)
eps = 0.0 # controls convergence criterion

# begin simulation
# begin simulation

# Tip0: Use tf.function --> helps in speeding up the training
# Example

#@tf.function
#def regress(X, theta):
#	# WRITEME: write your code here to complete the routine
#	# Define your forward pass
#	return -1.0


def sigmoid(z):
	result = 1/(1+np.exp(-z))
	return result
	# WRITEME: write your code here to complete the routine
    

def predict(X, theta):
    
	# WRITEME: write your code here to complete the routine
	result = sigmoid(regress(X,theta))>0.5
	return result
	
def regress(X, theta):
	# WRITEME: write your code here to complete the routine
	# Define your forward pass
	result = theta[0] + np.matmul(X,theta[1].transpose())
	return result


def gaussian_log_likelihood(mu, y):
	# WRITEME: write your code here to complete the sub-routine
	# Define loss function
	term_1 = np.multiply(mu,y)
	term_2 = np.multiply((1-y),(1-mu))
	result = np.sum(term_1+term_2)
	return result
	
	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
	# WRITEME: write your code here to complete the routine
	# Cost function is also known as loss function 
	m = len(y)
	a = (beta/(2*m))*np.sum(np.power(theta[1],2))
	fun = regress(X,theta)
    
	result = (-1/(m))*gaussian_log_likelihood(fun,y)+a
	return result

def computeGrad(X, y, theta, beta): 
	# WRITEME: write your code here to complete the routine
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
	m = len(y)
	fun = regress(X,theta)
	dL_dfy = -(1/m)*np.sum(X) # derivative w.r.t. to model output units (fy)
	dL_db = (1/m)*np.sum(fun-y) # derivative w.r.t. model weights w
	dL_dw = (1/m)*np.sum(np.multiply(fun-y,X),axis=0)# derivative w.r.t model bias b
	dL_dw_reg = (beta/m)*theta[1]
	nabla = (dL_db, dL_dw+dL_dw_reg) # nabla represents the full gradient
	# You can also use gradient tape and replace this function
	return nabla
	
path = os.getcwd() + '/data/prob3.dat'  
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
# Tip1: Convert .dat into numpy and use tensor flow api to process data
# display some information about the dataset itself here
# WRITEME: write your code here to print out information/statistics about the data-set "data" 

# WRITEME: write your code here to create a simple scatterplot of the dataset itself and print/save to disk the result




positive = data2[data2['Accepted'].isin([1])]  
negative = data2[data2['Accepted'].isin([0])]


#TODO
#Convert positive and negative samples into tf.Variable 
x1 = data2['Test 1']  
x2 = data2['Test 2']
#Convert x1 and x2 to tensorflow variables
# apply feature map to input features x1 and x2
cnt = 0
for i in range(1, degree+1):  
	for j in range(0, i+1):
		data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
		cnt += 1

data2.drop('Test 1', axis=1, inplace=True)  
data2.drop('Test 2', axis=1, inplace=True)

# set X and y
cols = data2.shape[1]
rows = data2.shape[0]  
X2 = data2.iloc[:,1:cols]  
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)  
y2 = np.array(y2.values)  
w = np.zeros((1,X2.shape[1]))
b = np.array([0])
theta = (b, w)
### Important please read all comments
### or use tf.Variable to define w and b if using Keras and gradient tape)
theta = (b, w)

L = computeCost(X2, y2, theta,beta)
print("-1 L = {0}".format(L))
L_best = L
i = 0
cost = [] # you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
#Initialize graph and all variables
while(i < n_epoch):
	dL_db, dL_dw = computeGrad(X2, y2, theta, beta)
	b = theta[0]
	w = theta[1]
	# update rules go here...
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
	b = b-(alpha*dL_db)
	w = w-(alpha*dL_dw)
	theta = (b,w)
	# (note: don't forget to override the theta variable...)
	L = computeCost(X2, y2, theta,beta) # track our loss after performing a single step
	# Use function 
	score = np.sum(predict(X2,theta)==y2)/rows

	print("For the epoch %i -- L = %f  -- b = %f -- score = %f --  "%(i,L,b,score))
	print("the values of w are as follows")
	print(w)
	i += 1

	cost.append(L)
    
	if len(cost)>1:
	    if (abs(cost[-1] - cost[-2]) < eps):
	        break
    
	# print parameter values found after the search
	#print W
#print b
#Save everything into saver object in tensorflow
#Visualize using tensorboard
kludge = 0.25 # helps with printing the plots (you can tweak this value if you like)
# visualize the fit against the data

# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)

print(" \t\t alpha:{0}, eps:{1}, beta:{2}".format(alpha,eps,beta)) # plot detaills


h = .01  # step size in the mesh
# create a mesh to plot in
x_min, x_max = x1.min() - 1, x1.max() + 1
y_min, y_max = x2.min() - 1, x2.max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
grid = np.hstack((r1,r2))

grid_nl = []
for i in range(1, degree+1):  
	for j in range(0, i+1):
		feat = np.power(r1, i-j) * np.power(r2, j)
		if (len(grid_nl) > 0):
			grid_nl = np.c_[grid_nl, feat]
		else:
			grid_nl = feat


yhat = predict(grid_nl,theta)
zz = yhat.reshape(xx.shape)
plt.contourf(xx, yy, zz, cmap='Paired')
plt.scatter(negative['Test 1'],negative['Test 2'],color='green',label="Negative")
plt.scatter(positive['Test 1'],positive['Test 2'],color='red',label="Positive")
plt.legend(loc="best")
plt.show()
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].


# visualize the loss as a function of passes through the dataset
# WRITEME: write your code here create and save a plot of loss versus epoch
plt.figure()
plt.title("loss against epochs") 
plt.scatter(range(len(cost)),cost,color='black',label="loss")         
plt.legend(loc="best")
plt.show() # convenience command to force plots to pop up on desktop
print("\t\t alpha:{0}, eps:{1}, beta:{2}".format(alpha,eps,beta)) # plot detaills
# Get final probabilities	 --> you can replace this with your Keras predict (if eager or math is difficult to process)
