package com.nickolas.wood.neural_network.simple;

import java.util.Random;

import static com.nickolas.wood.neural_network.utils.Utils.*;

public class NeuralNetworkSimple {

    final private double[][] trainingInputs = new double[][]{{0.0, 0.0, 1.0},
            {1.0, 1.0, 1.0},
            {1.0, 0.0, 1.0},
            {0.0, 1.0, 1.0}};

    final private double[] trainingOutputs = new double[]{0.0, 1.0, 1.0, 0.0};

    final private Random random = new Random();

    private double[] synapticWeights = new double[] {
            (2 * random.nextDouble()) - 1,
            (2 * random.nextDouble()) - 1,
            (2 * random.nextDouble()) - 1
    };

    public NeuralNetworkSimple() {

        double[] outputLayer = new double[4];

        for (int i = 0; i < 10000000; i++) {

            double[][] input_layer = trainingInputs;

            double[] dotProdLayer = DotProd(input_layer, synapticWeights);

            for (int outputLayerIndex = 0; outputLayerIndex < dotProdLayer.length; outputLayerIndex++) {

                outputLayer[outputLayerIndex] = Sigmoid(dotProdLayer[outputLayerIndex], sigmoidDouble);

            }

            double[] adjustments = new double[4];

            for (int outputLayerIndex = 0; outputLayerIndex < outputLayer.length; outputLayerIndex++) {

                adjustments[outputLayerIndex] = ErrorWeightedDerivative(
                        outputLayer[outputLayerIndex], trainingOutputs[outputLayerIndex]);

            }

            synapticWeights = VectorAdd(synapticWeights, DotProd(input_layer, adjustments));

        }



        for (double v : outputLayer) {

            System.out.println(v);

        }

    }
}
