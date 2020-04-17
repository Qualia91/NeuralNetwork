package com.nick.wood.neural_network.utils;

public class Utils {

    public static final Function<Double> sigmoidDouble = x -> 1.0 / (1.0 + Math.exp(-x));
    public static final Function<Float> sigmoidFloat = x -> 1.0f / (1.0f + (float) Math.exp(-x));

    interface Function<T> {
        T apply(T x);
    }

    //normalising function (Sigmoid function)
    public static double Sigmoid(double x) {

        return 1.0 / (1.0 + Math.exp(-x));

    }

    //normalising function (Sigmoid function)
    public static <T> T Sigmoid(T x, Function<T> func) {

        return func.apply(x);

    }

    //normalising function (Sigmoid function) vector form
    public static double[] Sigmoid(double[] x) {

        double[] returnVector = new double[x.length];

        for (int i = 0; i < x.length; i++) {

            returnVector[i] = Sigmoid(x[i]);

        }

        return returnVector;

    }

    //normalising function (Sigmoid function) vector form
    public static double[][] Sigmoid(double[][] x) {

        double[][] returnVector = new double[x.length][];

        for (int row = 0; row < x.length; row++) {

            returnVector[row] = Sigmoid(x[row]);

        }

        return returnVector;

    }

    // transpose
    public static double[][] Transpose(double[][] inputMatrix) {

        double[][] outputMatrix = new double[inputMatrix[0].length][inputMatrix.length];

        for (int rowIndex = 0; rowIndex < inputMatrix.length; rowIndex++) {

            for (int colIndex = 0; colIndex < inputMatrix[rowIndex].length; colIndex++) {

                outputMatrix[colIndex][rowIndex] = inputMatrix[rowIndex][colIndex];

            }

        }

        return outputMatrix;

    }

    // transpose
    public static double[][] Transpose(double[] inputMatrix) {

        double[][] outputMatrix = new double[inputMatrix.length][1];

        for (int rowIndex = 0; rowIndex < inputMatrix.length; rowIndex++) {

            outputMatrix[rowIndex][0] = inputMatrix[rowIndex];

        }

        return outputMatrix;

    }

    public static double[][] MatrixAdd(double[][] a, double[][] b) {

        double[][] newMatrix = new double[a.length][a[0].length];

        for (int rowIndex = 0; rowIndex < newMatrix.length; rowIndex++) {

            for (int colIndex = 0; colIndex < a[rowIndex].length; colIndex++) {

                newMatrix[rowIndex][colIndex] = a[rowIndex][colIndex] + b[rowIndex][colIndex];

            }

        }

        return newMatrix;

    }

    public static double[][] MatrixAdd(double[][] a, double[] b) {

        double[][] newMatrix = new double[a.length][a[0].length];

        for (int rowIndex = 0; rowIndex < newMatrix.length; rowIndex++) {

            for (int colIndex = 0; colIndex < a[rowIndex].length; colIndex++) {

                newMatrix[rowIndex][colIndex] = a[rowIndex][colIndex] + b[rowIndex];

            }

        }

        return newMatrix;

    }

    public static double[][] Subtract(double[][] a, double[][] b) {

        double[][] newMatrix = new double[a.length][a[0].length];

        for (int rowIndex = 0; rowIndex < newMatrix.length; rowIndex++) {

            for (int colIndex = 0; colIndex < a[rowIndex].length; colIndex++) {

                newMatrix[rowIndex][colIndex] = a[rowIndex][colIndex] - b[rowIndex][colIndex];

            }

        }

        return newMatrix;

    }

    public static double[] VectorAdd(double[] a, double[] b) {

        double[] newMatrix = new double[a.length];

        for (int i = 0; i < newMatrix.length; i++) {

            newMatrix[i] = a[i] + b[i];

        }

        return newMatrix;

    }

    public static double[] VectorMultiply(double[] a, double[] b) {

        double[] newVector = new double[a.length];

        for (int i = 0; i < newVector.length; i++) {

            newVector[i] = a[i] * b[i];

        }

        return newVector;

    }

    public static double[][] MatrixMultiply(double[][] a, double[][] b) {

        double[][] newMatrix = new double[a.length][];

        for (int i = 0; i < newMatrix.length; i++) {

            newMatrix[i] = VectorMultiply(a[i], b[i]);

        }

        return newMatrix;

    }

    public static double[] DotProd(double[][] matrix, double[] vector) {

        double[] output = new double[matrix.length];

        for (int inputRowIndex = 0; inputRowIndex < matrix.length; inputRowIndex++) {

            double sum = 0;

            for (int inputColIndex = 0; inputColIndex < matrix[inputRowIndex].length; inputColIndex++) {

                sum += matrix[inputRowIndex][inputColIndex] * vector[inputColIndex];

            }

            output[inputRowIndex] = sum;

        }

        return output;

    }

    public static double DotProd(double[] vectorOne, double[] vectorTwo) {

        double sum = 0;

        for (int inputRowIndex = 0; inputRowIndex < vectorOne.length; inputRowIndex++) {

            sum += vectorOne[inputRowIndex] * vectorTwo[inputRowIndex];

        }

        return sum;

    }

    // TODO: need to have this transpose matrixTwo before using it so it implements the dot product correctly
    public static double[][] DotProd(double[][] matrixOne, double[][] matrixTwo) {

        double[][] returnMatrix = new double[matrixOne.length][matrixTwo.length];

        for (int matrixOneRow = 0; matrixOneRow < matrixOne.length; matrixOneRow++) {

            for (int matrixTwoRow = 0; matrixTwoRow < matrixTwo.length; matrixTwoRow++) {

                double sum = 0;

                for (int matrixOneCol = 0; matrixOneCol < matrixOne[matrixOneRow].length; matrixOneCol++) {

                    sum += matrixOne[matrixOneRow][matrixOneCol] * matrixTwo[matrixTwoRow][matrixOneCol];

                }

                returnMatrix[matrixOneRow][matrixTwoRow] = sum;

            }

        }

        return returnMatrix;

    }


    public static double[][] Multiply(double[][] matrixOne, double value) {

        double[][] returnMatrix = new double[matrixOne.length][matrixOne[0].length];

        for (int matrixOneRow = 0; matrixOneRow < matrixOne.length; matrixOneRow++) {

            for (int matrixOneCol = 0; matrixOneCol < matrixOne[matrixOneRow].length; matrixOneCol++) {

                returnMatrix[matrixOneRow][matrixOneCol] = matrixOne[matrixOneRow][matrixOneCol] * value;

            }


        }

        return returnMatrix;

    }

    // sigmoid derivative
    public static double SigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public static double[] SigmoidDerivative(double[] x) {

        double[] returnX = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            returnX[i] = SigmoidDerivative(x[i]);
        }

        return returnX;

    }

    public static double[][] SigmoidDerivative(double[][] x) {

        double[][] returnX = new double[x.length][];

        for (int i = 0; i < x.length; i++) {
            returnX[i] = SigmoidDerivative(x[i]);
        }

        return returnX;

    }

    // error weighted derivative
    public static double ErrorWeightedDerivative(double predictedValue, double actualValue) {

        double valueError = actualValue - predictedValue;

        return valueError * SigmoidDerivative(predictedValue);

    }

    public static double[] copy(double[] vector) {

        double[] returnVector = new double[vector.length];

        System.arraycopy(vector, 0, returnVector, 0, vector.length);

        return returnVector;
    }

    public static double[][] copy(double[][] matrix) {

        double[][] returnMatrix = new double[matrix.length][];

        for (int row = 0; row < matrix.length; row++) {
            returnMatrix[row] = copy(matrix[row]);
        }

        return returnMatrix;
    }

    public static double[][][] copy(double[][][] m) {
        double[][][] copy = new double[m.length][][];

        for (int index = 0; index < m.length; index++) {
            copy[index] = copy(m[index]);
        }

        return copy;
    }

    public static void printMatrix(double[][] cost) {
        for (int inputIndex = 0; inputIndex < cost.length; inputIndex++) {
            StringBuilder output = new StringBuilder("Input row: " + inputIndex + " values: ");
            for (int value = 0; value < cost[inputIndex].length; value++) {
                output.append(cost[inputIndex][value]);
                output.append(" ");
            }
            System.out.println(output);
        }
    }

    public static void printVector(double[] data) {
        StringBuilder output = new StringBuilder("values: ");
        for (double datum : data) {
            output.append(datum);
            output.append(" ");
        }
        System.out.println(output);
    }
}
