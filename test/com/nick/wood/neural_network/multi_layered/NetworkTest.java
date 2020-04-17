package com.nick.wood.neural_network.multi_layered;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NetworkTest {

    @Test
    public void networkWalkthroughTest() {


        double[][] trainingInputs = new double[][]{
                {0.05, 0.10}
        };

        double[][] trainingOutputs = new double[][]{{0.01, 0.99}};

        int[] sizes = new int[]{2, 2};

        double[][][] weights = new double[][][]{
                // layer one
                {
                        {0.15, 0.25},
                        {0.2, 0.3},
                },
                // layer two
                {
                        {0.40, 0.5},
                        {0.45, 0.55}
                }
        };

        double[][] biases = new double[][]{
                // layer one
                {0.35, 0.35},
                // layer two
                {0.6, 0.6},
        };

        Network n = new Network(
                sizes,
                weights,
                biases
        );

        double[][] trainedAnswers = n.trainData(1, trainingInputs, trainingOutputs, sizes);

        assertEquals(0.75136507, trainedAnswers[0][0], 0.00000001);
        assertEquals(0.772928465, trainedAnswers[0][1], 0.00000001);

        double[][] trainedAnswers2 = n.trainData(10000, trainingInputs, trainingOutputs, sizes);

        assertEquals(0.01, trainedAnswers2[0][0], 0.01);
        assertEquals(0.99, trainedAnswers2[0][1], 0.01);

    }

    @Test
    public void networkWalkthroughWithRandomStartsTest() {


        double[][] trainingInputs = new double[][]{
                {0.05, 0.10}
        };

        double[][] trainingOutputs = new double[][]{{0.01, 0.99}};

        int[] sizes = new int[]{2};

        Network n = new Network();

        double[][] trainedAnswers = n.trainData(100000, trainingInputs, trainingOutputs, sizes);

        assertEquals(0.010, trainedAnswers[0][0], 0.001);
        assertEquals(0.989, trainedAnswers[0][1], 0.001);

    }

    @Test
    public void networkWalkthroughWithRandomStartsLowerNumberOfNeuronsTest() {


        double[][] trainingInputs = new double[][]{
                {0.05, 0.10}
        };

        double[][] trainingOutputs = new double[][]{{0.01, 0.99}};

        int[] sizes = new int[]{1};

        Network n = new Network();

        double[][] trainedAnswers = n.trainData(100000, trainingInputs, trainingOutputs, sizes);

        assertEquals(0.010, trainedAnswers[0][0], 0.001);
        assertEquals(0.989, trainedAnswers[0][1], 0.001);

    }

    @Test
    public void networkWalkthroughWithRandomStartsAndExtraNeuronsInHiddenLayerTest() {


        double[][] trainingInputs = new double[][]{
                {0.05, 0.10}
        };

        double[][] trainingOutputs = new double[][]{{0.01, 0.99}};

        int[] sizes = new int[]{10};

        Network n = new Network();

        double[][] trainedAnswers = n.trainData(100000, trainingInputs, trainingOutputs, sizes);

        assertEquals(0.010, trainedAnswers[0][0], 0.001);
        assertEquals(0.989, trainedAnswers[0][1], 0.001);

    }

    @Test
    public void networkDifferentTest() {

        double[][] trainingInputs = new double[][]{{0.0, 0.0, 1.0},
                {1.0, 1.0, 1.0},
                {1.0, 0.0, 1.0},
                {0.0, 1.0, 1.0}};

        double[][] trainingOutputs = new double[][]{{0.0},
                {1.0},
                {1.0},
                {0.0}};

        for (int i = 0; i < 10; i++) {

            double minCost = Double.MAX_VALUE;
            int minCostNoN = Integer.MAX_VALUE;
            int minCostIterations = Integer.MAX_VALUE;
            for (int numberOfNeurons = 3; numberOfNeurons < 20; numberOfNeurons++) {
                for (int iterations = 1; iterations < 100000; iterations *= 10) {
                    int[] sizes = new int[]{numberOfNeurons};

                    Network n = new Network();

                    double[][] trainedAnswers = n.trainData(iterations, trainingInputs, trainingOutputs, sizes);

                    for (int answerIndex = 0; answerIndex < trainedAnswers.length; answerIndex++) {

                        double cost = Math.abs(trainingOutputs[answerIndex][0] - trainedAnswers[answerIndex][0]);

                        if (cost < minCost) {
                            minCost = cost;
                            minCostNoN = numberOfNeurons;
                            minCostIterations = iterations;
                        }

                    }

                }

            }

            System.out.println("Number of neurons: " + minCostNoN + " iterations: " + minCostIterations + " Error: " + minCost);

        }
    }


    @Test
    public void networkDiffOutputBiggerThanOneTest() {

        // have to scale them to between. need to play around with other activation functions that don't go from o to 1 like sigmoid

        double[][] trainingInputs = new double[][]{{0.03, 0.05},
                {0.05, 0.01},
                {0.1, 0.02}};

        double[][] trainingOutputs = new double[][]{{0.75},
                {0.82},
                {0.93}};

        int[] sizes = new int[]{8};

        Network n = new Network(
        );

        double[][] outputs = n.trainData(500000, trainingInputs, trainingOutputs, sizes);

        assertEquals(0.75, outputs[0][0], 0.001);
        assertEquals(0.82, outputs[1][0], 0.001);
        assertEquals(0.93, outputs[2][0], 0.001);

        System.out.println();
    }

    @Test
    public void additionTest() {

        double[][][] testData = createTestData();
        double[][] trainingInputs = testData[0];
        double[][] trainingOutputs = testData[1];

        int[] sizes = new int[]{3};

        Network n = new Network();

        double[][] outputs = n.trainData(5000, trainingInputs, trainingOutputs, sizes);

        System.out.println();

        double[] newInputs = new double[] { 0.143, 0.169};

        double[][][] feedForwardOutputs = n.feedForward(newInputs);

        System.out.println();


    }

    private static double[][][] createTestData() {

        Random rand = new Random();

        double[][][] output = new double[2][][];

        double[][] testData = new double[2000][2];
        double[][] testOutput = new double[2000][1];

        for (int row = 0; row < testData.length; row++) {
            for (int col = 0; col < testData[row].length; col++) {

                double firstNumber = rand.nextInt(20);
                double secondNumber = rand.nextInt(20);

                testData[row][0] = firstNumber/100;
                testData[row][1] = secondNumber/100;

                testOutput[row][0] = (firstNumber + secondNumber) / 100;
            }
        }

        output[0] = testData;
        output[1] = testOutput;

        return output;

    }


}