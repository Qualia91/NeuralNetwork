package com.nickolas.wood.neural_network.simple;

import java.util.Random;

import static com.nickolas.wood.neural_network.utils.Utils.*;

public class NeuralNetworkSimpleGeneral {

    final private double[][] trainingInputs;

    final private double[] trainingOutputs;

    private double[] synapticWeights;

    public NeuralNetworkSimpleGeneral(double[][] trainingInputs, double[] trainingOutputs) {

        this.trainingInputs = trainingInputs;
        this.trainingOutputs = trainingOutputs;
        synapticWeights = new double[trainingOutputs.length];

        Random random = new Random();

        for (int outputIndex = 0; outputIndex < trainingOutputs.length; outputIndex++) {

            synapticWeights[outputIndex] = (2 * random.nextDouble()) - 1;

        }

    }

    public void train(double iterations) {

        double[] outputLayer = new double[trainingOutputs.length];

        for (int i = 0; i < iterations; i++) {

            double[][] input_layer = this.trainingInputs;

            double[] dotProdLayer = DotProd(input_layer, synapticWeights);

            for (int outputLayerIndex = 0; outputLayerIndex < dotProdLayer.length; outputLayerIndex++) {

                outputLayer[outputLayerIndex] = Sigmoid(dotProdLayer[outputLayerIndex], sigmoidDouble);

            }

            double[] adjustments = new double[trainingOutputs.length];

            for (int outputLayerIndex = 0; outputLayerIndex < outputLayer.length; outputLayerIndex++) {

                adjustments[outputLayerIndex] = ErrorWeightedDerivative(
                        outputLayer[outputLayerIndex], this.trainingOutputs[outputLayerIndex]);

            }

            synapticWeights = VectorAdd(synapticWeights, DotProd(input_layer, adjustments));

        }

        for (double v : outputLayer) {

            System.out.println(v);

        }

    }

    public double evaluate(double[] newInputs) {

        double dotProd = DotProd(newInputs, synapticWeights);

        return Sigmoid(dotProd, sigmoidDouble);

    }

}
