#!/usr/bin/env python

import sys
sys.path.insert(0, '/home/mackenzie/NICE/cpp/interface')
import Nice4Py
import numpy as np

#ALWAYS CHECK THE TYPE IF IM GETTING A FUNKY OUTPUT

matrixA = np.matrix('2.0, 3.0; 4.0, 5.0', dtype=np.float32)
matrixB = np.matrix('6.0, 7.0 ; 8.0, 9.0', dtype=np.float32)
cpu_op = Nice4Py.CPUOp()

#CHECKS MULTIPLYMATRIX FUNCTIONS IN NICE
# matrixC = np.matrix('0, 0 ; 0, 0', dtype=np.float32)
# matrixD = np.matrix('0, 0 ; 0, 0', dtype=np.float32)
# scalar = 2.0
# print matrixA
# print matrixB
# print "This is the product of matrix*matrix from numpy multiplication: ", "\n", matrixA*matrixB #36,41,64,73
# print "The is the product of matrix*scalar from numpy multiplication: ", "\n", matrixA*scalar #4,6,8,10
# cpu_op = Nice4Py.CPUOp()
# cpu_op.MultiplyMatrix(matrixA, 2, 2, matrixB, 2, 2, matrixC)
# print "This is the product from NICE: ", "\n", matrixC
# cpu_op.MultiplyMatrix(matrixA, 2, 2, matrixD, scalar)
# print "This is the product of matrix*scalar from NICE: ", "\n", matrixD

#CHECKS INVERSEMATRIX FUNCTION
# matrixEmpty = np.matrix('0, 0 ; 0, 0', dtype=np.float32)
# print "Here is the matrix A: ", "\n", matrixA
# cpu_op = Nice4Py.CPUOp()
# cpu_op.InverseMatrix(matrixA, 2, 2, matrixEmpty)
# print "Here is the matrix A inverse: ", "\n", matrixEmpty

#CHECKS NORMMATRIX FUNCTION
# cpu_op = Nice4Py.CPUOp()
# vectorA = np.matrix('0; 0', dtype=np.float32)
# vectorB = np.matrix('0; 0', dtype=np.float32)
# cpu_op.NormMatrix(matrixA, 2, 2, 2, 0, vectorA)
# cpu_op.NormMatrix(matrixB, 2, 2, 2, 1, vectorB)
# print "The original matrixA is ", "\n", matrixA
# print "The norm vectorA is ", "\n", vectorA
# print "The original matrixB is ", "\n", matrixB
# print "The norm vectorA is ", "\n", vectorB

#CHECKS CENTERMATRIX FUNCTION
# matrixZ = np.matrix('0, 0; 0, 0', dtype=np.float32)
# print matrixB
# cpu_op.CenterMatrix(matrixB, 2, 2, 1, matrixZ)
# print matrixZ

#CHECKS NORMALIZE MATRIX FUNCTION
# matrixY = np.matrix('0, 0; 0, 0', dtype=np.float32)
# print "This is the matrix: ", "\n", matrixA
# cpu_op.NormalizeMatrix(matrixA, 2, 2, 2, 0, matrixY)
# print "The is the normalized matrix: ", "\n", matrixY

#CHECKS ST DEV FUNCTION
matrixSTB = np.matrix('0; 0', dtype=np.float32)
print "Here is the matrix: ", "\n", matrixB
cpu_op.StandardDeviationMatrix(matrixA, 2, 2, 0, matrixSTB)
print "Here is the standard deviations of the matrix: ", "\n", matrixSTB