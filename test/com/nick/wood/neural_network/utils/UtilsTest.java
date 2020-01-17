package com.nick.wood.neural_network.utils;

import org.junit.Test;

import static org.junit.Assert.*;

public class UtilsTest {

    @Test
    public void sigmoidSingleDigitUnitTest() {

        double returnValue = Utils.Sigmoid(1);

        double answer = 0.73105857863000487925115924182182;

        assertEquals("Sigmoid Single Digit test", answer, returnValue, 0.000000000001);

    }

    @Test
    public void sigmoidVectorUnitTest() {

        double[] returnValue = Utils.Sigmoid(new double[]{1, 1, 1});

        double[] answer = new double[] {0.73105857863000487925115924182182,
                0.73105857863000487925115924182182,
                0.73105857863000487925115924182182};

        for (int i = 0; i < returnValue.length; i++) {

            assertEquals("Sigmoid Vector test, index " + i, answer[i], returnValue[i], 0.000000000001);

        }

    }

    @Test
    public void transposeMatrix() {

        double[][] input = new double[][] {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0}
        };

        double[][] expected = new double[][] {
                {1.0, 4.0, 7.0},
                {2.0, 5.0, 8.0},
                {3.0, 6.0, 9.0}
        };

        double[][] output = Utils.Transpose(input);

        HelperFunctions.MatrixAssert(expected, output);
    }

    @Test
    public void transposeVector() {

        double[] input = new double[] {1.0, 2.0, 3.0};

        double[][] expected = new double[][] {
                {1.0},
                {2.0},
                {3.0}
        };

        double[][] output = Utils.Transpose(input);

        HelperFunctions.MatrixAssert(expected, output);
    }

    @Test
    public void vectorAdd() {

        double[] input1 = new double[] {1.0, 2.0, 3.0};
        double[] input2 = new double[] {1.0, 2.0, 3.0};
        double[] expected = new double[] {2.0, 4.0, 6.0};

        double[] output = Utils.VectorAdd(input1, input2);

        HelperFunctions.VectorAssert(expected, output);

    }

    @Test
    public void matrixAdd() {

        double[][] input1 = new double[][] {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0}
        };

        double[][] input2 = new double[][] {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0}
        };

        double[][] expected = new double[][] {
                {2.0, 4.0, 6.0},
                {8.0, 10.0, 12.0},
                {14.0, 16.0, 18.0}
        };

        double[][] output = Utils.MatrixAdd(input1, input2);

        HelperFunctions.MatrixAssert(expected, output);

    }

    @Test
    public void multiplyTest() {

        double[][] input1 = new double[][] {
                {3.0, 5.0},
                {5.0, 1.0},
                {10.0, 2.0}
        };

        double[][] input2 = new double[][] {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        };

        double[][] expected = new double[][] {
                {(3 * 1 ) + (5 * 4), (3 * 2)  + (5 * 5), (3 * 3) +  (5 * 6)},
                {(5 * 1 ) + (1 * 4), (5 * 2)  + (1 * 5), (5 * 3) +  (1 * 6)},
                {(10 * 1) + (2 * 4), (10 * 2) + (2 * 5), (10 * 3) + (2 * 6)}
        };

        double[][] actual = Utils.DotProd(input1, input2);

        HelperFunctions.MatrixAssert(expected,actual);

    }

    @Test
    public void dotProdMatrixVector() {

        double[][] input1 = new double[][] {
                {3.0, 5.0, 4.0},
                {5.0, 1.0, 3.0},
                {10.0, 2.0, 1.0}
        };

        double[] input2 = new double[] {
                1.0, 2.0, 3.0
        };

        double[] expected = new double[] {
                (3 * 1) + (5 * 2) + (4 * 3),
                (5 * 1) + (1 * 2) + (3 * 3),
                (10 * 1) + (2 * 2) + (1 * 3)
        };

        double[] actual = Utils.DotProd(input1, input2);

        HelperFunctions.VectorAssert(expected,actual);

    }

    @Test
    public void sigmoidDerivative() {
    }

    @Test
    public void errorWeightedDerivative() {
    }
}