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
trial_name = 'p1_fit' # will add a unique sub-string to output of this program
degree = 12 # p, order of model
beta = 0.0001 # regularization coefficient
alpha = 0.09 # step size coefficient
eps = 0.000001 # controls convergence criterion
n_epoch = 15000 # number of epochs (full passes through the dataset)

# begin simulation

# Tip0: Use tf.function --> helps in speeding up the training
# Example

#@tf.function
#def regress(X, theta):
#	# WRITEME: write your code here to complete the routine
#	# Define your forward pass
#	return -1.0

def regress(X, theta):
	# WRITEME: write your code here to complete the routine
	# Define your forward pass
	result = theta[0] + np.matmul(X,theta[1].transpose())
	return result

def gaussian_log_likelihood(mu, y):
	# WRITEME: write your code here to complete the sub-routine
	# Define loss function
	result = np.sum(np.power(mu-y,2))
	return result
	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
	# WRITEME: write your code here to complete the routine
	# Cost function is also known as loss function 
#    import pdb
	m = len(y)
#    import pdb
#    pdb.set_trace()
	fun = regress(X,theta)
	a = (beta/(2*m))*np.sum(np.power(theta[1],2))
	result = (1/(2*m))*gaussian_log_likelihood(fun,y)+a
    # function call from above for gaussian log likelihood
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


path = os.getcwd() + '/data/prob2.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y']) 
# Tip1: Convert .dat into numpy and use tensor flow api to process data
# display some information about the dataset itself here
# WRITEME: write your code here to print out information/statistics about the data-set "data" 
print("Number of samples: {}".format(data.shape[0]))
print("Dimension of sample: {}".format(data.shape[1]-1))

# WRITEME: write your code here to create a simple scatterplot of the dataset itself and print/save to disk the result
plt.figure()
plt.title("Sample Scatter plot of Data")
plt.scatter(data.iloc[:,0],data.iloc[:,1])         
#plt.scatter(X[:,0],regress(X,theta))
plt.show()


# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
    
#plt.figure()
#plt.scatter(X,y)         
#plt.show()

# convert from data frames to numpy matrices
X = np.array(X.values)  
for d in range(degree-1):
    X = np.concatenate((X,np.power(X[:,0:1],d+2)),axis = 1)

y = np.array(y.values)

#TODO convert np array to tensor objects if working with Keras
# convert to numpy arrays and initalize the parameter array theta 
w = np.zeros((1,X.shape[1]))
b = np.array([0])
theta = (b, w)
### Important please read all comments
### or use tf.Variable to define w and b if using Keras and gradient tape
L = computeCost(X, y, theta,beta)
print("-1 L = {0}".format(L))
L_best = L
i = 0
cost = [] # you can use this list variable to help you create the loss versus epoch plot at the end (if you want)



while(i < n_epoch):
	dL_db, dL_dw = computeGrad(X, y, theta, beta)
	b = theta[0]
	w = theta[1]
	# update rules go here...
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
	b = b-(alpha*dL_db)
	w = w-(alpha*dL_dw)
	theta = (b,w)
	# (note: don't forget to override the theta variable...)
	L = computeCost(X, y, theta,beta) # track our loss after performing a single step
	# Use function 
	print("For the epoch %i -- L = %f  -- b = %f --   "%(i,L,b))
	print("the values of w are as follows")
	print(w)
	i += 1

	cost.append(L)
    
	if len(cost)>1:
	    if abs(cost[-1] - cost[-2]) < eps:
	        break
	# print parameter values found after the search
	#print W
    
    

#print b
#Save everything into saver object in tensorflow
#Visualize using tensorboard
kludge = 0.25 # helps with printing the plots (you can tweak this value if you like)
# visualize the fit against the data

# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)
plt.figure()
plt.title("fit against the data")
plt.scatter(X[:,0],y,label="raw data")         
plt.scatter(X[:,0],regress(X,theta),label="predicted")
plt.legend(loc="best")
plt.show()
print("\t\t alpha:{0}, eps:{1}, beta:{2}".format(alpha,eps,beta))
# visualize the loss as a function of passes through the dataset
# WRITEME: write your code here create and save a plot of loss versus epoch
plt.figure()
plt.title("loss against epochs")
plt.scatter(range(len(cost)),cost,label="loss")         
plt.legend(loc="best")
plt.show()
print("\t\t alpha:{0}, eps:{1}, beta:{2}".format(alpha,eps,beta))

# plt.show() # convenience command to force plots to pop up on desktop
