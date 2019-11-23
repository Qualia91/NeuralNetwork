package com.nickolas.wood.neural_network;

import com.nickolas.wood.neural_network.simple.NeuralNetworkSimpleGeneral;

public class NeuralNetworkSimpleGeneralTest {

    @org.junit.Test
    public void SimpleTest() {

        double[][] trainingInputs = new double[][]{{0.0, 0.0, 1.0},
                {0.0, 1.0, 0.0},
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 1.0},
                {1.0, 0.0, 1.0},
                {1.0, 1.0, 0.0},
                {1.0, 1.0, 1.0}};

        double[] trainingOutputs = new double[]{0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0};

        NeuralNetworkSimpleGeneral neuralNetworkSimpleGeneral = new NeuralNetworkSimpleGeneral(trainingInputs, trainingOutputs);
        neuralNetworkSimpleGeneral.train(1000000);
        System.out.println(neuralNetworkSimpleGeneral.evaluate(new double[] {0.0, 0.0, 0.0}));
    }

    @org.junit.Test
    public void SumRowTest() {

        double[][] trainingInputs = new double[][]{{0.0, 4.0, 1.0},
                {1.0, 99.0, 3.0},
                {1.0, 0.0, 55.55},
                {2.1, 1.0, 1.0},
                {0.0, 0.0, 0.0}};

        double[] trainingOutputs = new double[]{5.0, 103.0, 56.55, 4.1, 0.0};

        NeuralNetworkSimpleGeneral neuralNetworkSimpleGeneral = new NeuralNetworkSimpleGeneral(trainingInputs, trainingOutputs);
        neuralNetworkSimpleGeneral.train(1000000);
    }
}