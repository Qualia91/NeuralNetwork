package com.nick.wood.neural_network.utils;

import static org.junit.Assert.assertEquals;

public class HelperFunctions {

    public static void MatrixAssert(double[][] expected, double[][] actual) {
        for (int rowIndex = 0; rowIndex < expected.length; rowIndex++) {
            for (int colIndex = 0; colIndex < expected[rowIndex].length; colIndex++) {
                assertEquals("Matrix Test row index " + rowIndex + " col index " + colIndex,
                        expected[rowIndex][colIndex],
                        actual[rowIndex][colIndex],
                        0.000000000001);
            }
        }
    }

    public static void VectorAssert(double[] expected, double[] actual) {
        for (int index = 0; index < expected.length; index++) {
            assertEquals("Vector Test index " + index,
                    expected[index],
                    actual[index],
                    0.000000000001);
        }
    }


}
