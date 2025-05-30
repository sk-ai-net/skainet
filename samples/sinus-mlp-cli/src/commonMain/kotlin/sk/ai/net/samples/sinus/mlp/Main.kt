package sk.ai.net.samples.sinus.mlp

/**
 * Main entry point for the sinus MLP CLI application.
 *
 * This application loads weights from a GGUF or SafeTensors file and
 * approximates the value of sine using an MLP network with one input (angle),
 * 2 hidden layers, and 1 output neuron.
 */
expect fun main(args: Array<String>)