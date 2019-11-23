package com.nickolas.wood.neural_network;

import com.nickolas.wood.neural_network.multi_layered.Network;

import static com.nickolas.wood.neural_network.utils.Utils.printVector;

public class Main {

    public static void main(String[] args) {

        double[][] trainingInputs = new double[][]{{0.0, 0.0, 1.0},
                {1.0, 1.0, 1.0},
                {1.0, 0.0, 1.0},
                {0.0, 1.0, 1.0}};

        double[][] trainingOutputs = new double[][]{{0.0},
                {1.0},
                {1.0},
                {0.0}};

        int[] sizes = new int[]{3, 4, 2};

        Network n = new Network(
        );

        n.trainData(1000, trainingInputs, trainingOutputs, sizes);

        // feed forward
        double[] inputLayer = new double[]{
                0.0, 0.0, 0.0
        };

        for (int layerIndex = 0; layerIndex < sizes.length; layerIndex++) {

            inputLayer = n.calcNextFeedForward(inputLayer, layerIndex);
        }

        printVector(inputLayer);

    }

}
