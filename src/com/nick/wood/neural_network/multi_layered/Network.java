package com.nick.wood.neural_network.multi_layered;

import com.nick.wood.neural_network.utils.Utils;

import java.util.Arrays;
import java.util.Random;

public class Network {

    private final Random rand = new Random();

    // number of neurons in each layer
    private int[] sizes;

    // biases for each neuron in each layer
    private double[][] biases;

    // weights for each line from neuron to neuron for each neuron in each layer
    private double[][][] weights;

    public Network() {
    }

    public Network(int[] sizes, double[][][] weights, double[][] biases) {
        this.sizes = sizes;
        this.weights = weights;
        this.biases = biases;
    }

    public double[][][] feedForward(double[] trainingInputs) {

        double[][] allLayerData = new double[this.sizes.length + 1][];

        double[][] allLayerDataSigmoids = new double[this.sizes.length + 1][];

        double[] currentLayer = trainingInputs;

        double[] currentLayerSigmoids = currentLayer;

        allLayerData[0] = Utils.copy(currentLayer);
        allLayerDataSigmoids[0] = Utils.copy(currentLayerSigmoids);

        // feed forward
        for (int layerIndex = 0; layerIndex < this.sizes.length; layerIndex++) {

            currentLayer = Utils.VectorAdd(calcNextFeedForward(currentLayerSigmoids, layerIndex), biases[layerIndex]);
            currentLayerSigmoids = Utils.Sigmoid(currentLayer);

            allLayerData[layerIndex + 1] = Utils.copy(currentLayer);
            allLayerDataSigmoids[layerIndex + 1] = Utils.copy(currentLayerSigmoids);

        }

        double[][][] returnData = new double[2][][];
        returnData[0] = allLayerData;
        returnData[1] = allLayerDataSigmoids;
        return returnData;
    }

    public double[] calcNextFeedForward(double[] inputLayer, int layerIndex) {

        return Utils.DotProd(Utils.Transpose(weights[layerIndex]), inputLayer);

    }

    /**
     * Train neural network with input data
     *
     * @param iterations the number of iterations to run
     * @param trainingInputsArray input training data
     * @param trainingOutputsArray output training data
     * @param sizes description of neural layers
     */
    public double[][] trainData(int iterations, double[][] trainingInputsArray, double[][] trainingOutputsArray, int[] sizes) {

        if (weights == null) {

            this.sizes = new int[sizes.length + 1];
            System.arraycopy(sizes, 0, this.sizes, 0, sizes.length);
            this.sizes[this.sizes.length - 1] = trainingOutputsArray[0].length;

            weights = new double[this.sizes.length][][];

            // set weights from input data to first hidden layer
            weights[0] = new double[trainingInputsArray[0].length][sizes[0]];
            biases = new double[this.sizes.length][];
            for (int neuronIndex = 0; neuronIndex < weights[0].length; neuronIndex++) {

                for (int linkIndex = 0; linkIndex < weights[0][neuronIndex].length; linkIndex++) {

                    // gives gausian distributed random numbers between -1 and 1 with a mean of 0
                    weights[0][neuronIndex][linkIndex] = (2 * rand.nextDouble()) - 1;

                }

            }

            // set weights from hidden layers to the next
            for (int layerIndex = 0; layerIndex < this.sizes.length - 2; layerIndex++) {

                weights[layerIndex + 1] = new double[sizes[layerIndex]][sizes[layerIndex + 1]];

                for (int neuronIndex = 0; neuronIndex < weights[layerIndex].length; neuronIndex++) {

                    for (int linkIndex = 0; linkIndex < weights[layerIndex + 1][neuronIndex].length; linkIndex++) {

                        // gives gausian distributed random numbers between -1 and 1 with a mean of 0
                        weights[layerIndex + 1][neuronIndex][linkIndex] = (2 * rand.nextDouble()) - 1;

                    }

                }

            }

            // set weights for last hidden layer to output layer
            weights[weights.length - 1] = new double[sizes[sizes.length - 1]][trainingOutputsArray[0].length];
            for (int neuronIndex = 0; neuronIndex < weights[weights.length - 1].length; neuronIndex++) {

                for (int linkIndex = 0; linkIndex < weights[weights.length - 1][neuronIndex].length; linkIndex++) {

                    // gives gausian distributed random numbers between -1 and 1 with a mean of 0
                    weights[weights.length - 1][neuronIndex][linkIndex] = (2 * rand.nextDouble()) - 1;

                }

            }

            // set biases
            // set biases for each hidden layer
            for (int hiddenLayerIndex = 0; hiddenLayerIndex < sizes.length; hiddenLayerIndex++) {

                biases[hiddenLayerIndex] = new double[sizes[hiddenLayerIndex]];
                //Arrays.fill(biases[hiddenLayerIndex], (2 * rand.nextDouble()) - 1);
                Arrays.fill(biases[hiddenLayerIndex], 0.5);

            }

            // set biases for output layer
            biases[biases.length - 1] = new double[trainingOutputsArray[0].length];
            //Arrays.fill(biases[hiddenLayerIndex], (2 * rand.nextDouble()) - 1);
            Arrays.fill(biases[biases.length - 1], 0.5);

        }

        double[][] finalReturnedOutputs = Utils.copy(trainingOutputsArray);

        for (int i = 0; i < iterations; i++) {

            for (int trainingInputsIndex = 0; trainingInputsIndex < trainingInputsArray.length; trainingInputsIndex++) {

                double[] trainingInputs = trainingInputsArray[trainingInputsIndex];
                double[] trainingOutputs = trainingOutputsArray[trainingInputsIndex];

                // feed forward

                double[][][] feedForwardData = feedForward(trainingInputs);

                double[][] allLayerData = feedForwardData[0];
                double[][] allLayerDataSigmoids = feedForwardData[1];

                // FOR DEBUG
                //double[] errors = getErrors(currentLayerSigmoids, trainingOutputs);

                // back propagate error
                double[][][] updatedWeights = Utils.copy(weights);

                double[] dEtbdoutYValues = new double[trainingOutputs.length];
                double[] doutYbdYValues = new double[trainingOutputs.length];


                // from output to last hidden layer
                for (int firstNeuronIndex = 0; firstNeuronIndex < allLayerDataSigmoids[allLayerDataSigmoids.length - 2].length; firstNeuronIndex++) {

                    double dYbdw = allLayerDataSigmoids[allLayerDataSigmoids.length - 2][firstNeuronIndex];

                    for (int secondNeuronIndex = 0; secondNeuronIndex < trainingOutputs.length; secondNeuronIndex++) {

                        double dEtbdoutY = -(trainingOutputs[secondNeuronIndex] - allLayerDataSigmoids[allLayerDataSigmoids.length - 1][secondNeuronIndex]);

                        double doutYbdY = allLayerDataSigmoids[allLayerDataSigmoids.length - 1][secondNeuronIndex] * (1 - allLayerDataSigmoids[allLayerDataSigmoids.length - 1][secondNeuronIndex]);

                        dEtbdoutYValues[secondNeuronIndex] = dEtbdoutY;

                        doutYbdYValues[secondNeuronIndex] = doutYbdY;

                        double dEtbydW = dEtbdoutY * doutYbdY * dYbdw;

                        updatedWeights[updatedWeights.length - 1][firstNeuronIndex][secondNeuronIndex] = weights[updatedWeights.length - 1][firstNeuronIndex][secondNeuronIndex] - 0.5 * dEtbydW;

                    }

                }

                for (int firstNeuronIndex = 0; firstNeuronIndex < allLayerData[0].length; firstNeuronIndex++) {

                    for (int secondNeuronIndex = 0; secondNeuronIndex < updatedWeights[updatedWeights.length - 1].length; secondNeuronIndex++) {

                        double dEtbdoutH = 0;

                        for (int doutYbdYIndex = 0; doutYbdYIndex < doutYbdYValues.length; doutYbdYIndex++) {

                            dEtbdoutH += dEtbdoutYValues[doutYbdYIndex] * doutYbdYValues[doutYbdYIndex] * weights[1][secondNeuronIndex][doutYbdYIndex];

                        }

                        double doutHbdH = allLayerDataSigmoids[1][secondNeuronIndex] * (1 - allLayerDataSigmoids[1][secondNeuronIndex]);

                        updatedWeights[updatedWeights.length - 2][firstNeuronIndex][secondNeuronIndex] = weights[updatedWeights.length - 2][firstNeuronIndex][secondNeuronIndex] - (0.5 * dEtbdoutH * doutHbdH * allLayerData[0][firstNeuronIndex]);
                    }

                }

                weights = updatedWeights;

                finalReturnedOutputs[trainingInputsIndex] = Utils.copy(allLayerDataSigmoids[allLayerDataSigmoids.length - 1]);

            }

        }

        return finalReturnedOutputs;

    }

    private double[] getErrors(double[] lastLayerSigmoids, double[] targetOutputs) {

        double[] errors = new double[lastLayerSigmoids.length];

        for (int neuronIndex = 0; neuronIndex < lastLayerSigmoids.length; neuronIndex++) {

            errors[neuronIndex] = 0.5 * (targetOutputs[neuronIndex] - lastLayerSigmoids[neuronIndex]) * (targetOutputs[neuronIndex] - lastLayerSigmoids[neuronIndex]);

        }

        return errors;

    }

    private double[][] findCost(double[][] layerData, double[][] layerDataSigmoids, double[][] trainingOutputs) {

        return Utils.MatrixMultiply(Utils.Multiply(Utils.Subtract(trainingOutputs, layerDataSigmoids), -1), Utils.SigmoidDerivative(layerData));

    }

    public double[][][] getWeights() {

        return weights;
    }

    public void setWeights(double[][][] weights) {
        this.weights = weights;
    }
}
