import numpy as np
import math

def Nlargest_index(arr,N):
    #find indices of N largest element
    #https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
    arr=np.array(arr)
    return np.argpartition(arr, -N)[-N:]

def Nsmallest_index(arr,N):
    arr=np.array(arr)*-1
    return Nlargest_index(arr,N)

def sort_by_sum(a):
    #https://stackoverflow.com/questions/7235785/sorting-numpy-array-according-to-the-sum
    #https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
    return a[np.sum(a, axis = 1).argsort()]