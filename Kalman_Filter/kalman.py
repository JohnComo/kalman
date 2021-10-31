import numpy as np
from numpy.core.arrayprint import IntegerFormat 
from numpy.linalg import inv 

# This script will use a Kalman Filiter to track the center of a storm 

lattitude_coordinates = np.array([26.037, 26.132, 26.48, 26.32, 26.6])
longitude_coordinates = np.array([-88.76, -88.01, -87.4, -87, -87.2])

assert len(lattitude_coordinates) == len(longitude_coordinates), "VALUES MUST BE OF SAME LENGTH"

state_matrix = np.c_[lattitude_coordinates, longitude_coordinates]
 
#._c turns 1-d arrays into a 2-d stack 

# Initial Conditions 
acceleration = .1
velocity = 5
t = 1 # The difference in time between each of the lattitude and longitude observations

# Estimation Errors 
lattitude_estimated_error = .1
longitude_estimated_error = .1 

# Observation Errors 
lattitude_observed_error = .12 
longitude_observed_error = .12 


def predict(lat, long, t, a): 
    '''
    This function predicts where the center of the oil spill will be in the next time step 

    Params: 
    lat (array): array of lattitudes
    long (array): array of longitudes
    t(int): time_delta 
    a(float): accleration

    Returns: 
    A matrix X_prime represnting a distance given an acceleration
    '''
    # Updates previous state based on elapsed time
    A = np.array([[1,t],
                  [0,1]])

    # Current positions
    X = np.array([[lat], 
                   [long]])
    
    # Adjusts position position and velocity based on acceleration 
    B = np.array([[.5 * t **2], 
                    [t]])

    X_prime = A.dot(X) + B.dot(a)

    return X_prime 


def covariance(sigma1, sigma2): 
    '''
    This function returns updated covariance of estimations 

    Params: 
    sigma1 (float): estimated error of x 
    sigma2 (float): estimated error of y 

    Returns 
    Updated covariance matrix 
    '''
    cov1_2 = sigma1 * sigma2
    cov2_1 = sigma2 * sigma1

    cov = np.array([[sigma1 ** 2, cov1_2], 
                    [cov2_1, sigma2 ** 2]])

    update = np.diag(np.diag(cov))

    return update

# INITIAL ESTIMATION COVARIANCE MATRIX 
P = covariance(lattitude_estimated_error, longitude_estimated_error)
A = np.array([[1,t], 
             [0,1]])

# Initial State Matrix 
X = np.array([[state_matrix[0][0]], 
                [velocity]])

n = len(state_matrix[0])

for coord in state_matrix[1:]: 
    X = predict(X[0][0], X[1][0], t, acceleration)

    # diagnoals to 0 
    P = np.diag(np.diag(A.dot(P).dot(A.T)))

    # Kalman Gain - calculate the covariance matrix for observation errors and use it to compare with 
    # process covariance matrix. The diagnoal will be the weights that adjust observed position and vel

    I = np.identity(n)

    R = covariance(lattitude_observed_error, longitude_observed_error)

    S = I.dot(P).dot(I.T) + R 

    K = P.dot(I).dot(inv(S))

    # reshape 
    Y = I.dot(coord).reshape(n,-1)

    # Update the state matrix with combination of predicted state, measured values, cov matrix and Kalman Gain
    X = X + K.dot(Y - I.dot(X))

    # Update process covariance 
    P = (np.identity(len(K)) - K.dot(I)).dot(P)

    
print("Final location: ", X)

 # Unsure of how effective this will be because we are tryong to find a range of oil while 
 # most applications are being used to track vehicles, planes, etc

    

