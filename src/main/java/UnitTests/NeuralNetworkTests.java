package UnitTests;

import basicneuralnetwork.neuralnetwork.NeuralNetwork;
import org.ejml.simple.SimpleMatrix;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Classe NeuralNetworkTests: Testes unitários para a classe NeuralNetwork..
 * @author Márcio Felício, Maria Anjos, Miguel Rosa
 * @version 1.0 30/11/2024
 */
public class NeuralNetworkTests {

    /**
     * Testa a inicialização da rede neural para verificar os pesos e vieses.
     */
    @Test
    public void testInitialization() {
        NeuralNetwork nn = new NeuralNetwork(400, 2, 10, 1);

        assertNotNull(nn.getWeights());
        assertNotNull(nn.getBiases());
        assertEquals(400, nn.getInputNodes());
        assertEquals(10, nn.getHiddenNodes());
        assertEquals(1, nn.getOutputNodes());
    }

    /**
     * Testa o método `guess` para verificar se a saída está no intervalo esperado [0, 1].
     */
    @Test
    public void testGuessOutputRange() {
        NeuralNetwork nn = new NeuralNetwork(4, 2, 5, 1);
        nn.setActivationFunction("SIGMOID");

        double[] input = {0.5, 0.2, 0.8, 0.1};
        double[] output = nn.guess(input);

        assertNotNull(output);
        assertEquals(1, output.length);
        assertTrue(output[0] >= 0 && output[0] <= 1);
    }

    /**
     * Testa se o método `train` atualiza os pesos corretamente.
     */
    @Test
    public void testTrainUpdatesWeights() {
        NeuralNetwork nn = new NeuralNetwork(4, 1, 5, 1);
        nn.setActivationFunction("SIGMOID");
        nn.setLearningRate(0.01);

        double[] input = {0.1, 0.2, 0.3, 0.4};
        double[] target = {1.0};

        SimpleMatrix[] initialWeights = new SimpleMatrix[nn.getWeights().length];
        for (int i = 0; i < nn.getWeights().length; i++) {
            initialWeights[i] = nn.getWeights()[i].copy();
        }

        nn.train(input, target);

        boolean weightsChanged = false;
        SimpleMatrix[] updatedWeights = nn.getWeights();
        for (int i = 0; i < initialWeights.length; i++) {
            if (!initialWeights[i].isIdentical(updatedWeights[i], 1e-6)) {
                weightsChanged = true;
                break;
            }
        }

        assertTrue("Os pesos deveriam ter mudado após o treinamento", weightsChanged);
    }


    /**
     * Testa se a rede neural lança uma exceção para entradas de tamanho incorreto.
     */
    @Test(expected = RuntimeException.class)
    public void testInvalidInputSize() {
        NeuralNetwork nn = new NeuralNetwork(4, 1, 5, 1);
        double[] invalidInput = {0.1, 0.2}; // Tamanho incorreto

        nn.guess(invalidInput);
    }

    /**
     * Testa o método `copy` para verificar se a cópia é idêntica à original.
     */
    @Test
    public void testCopy() {
        NeuralNetwork original = new NeuralNetwork(4, 1, 5, 1);
        original.setActivationFunction("SIGMOID");

        NeuralNetwork copy = original.copy();

        assertNotNull(copy);
        assertArrayEquals(original.getDimensions(), copy.getDimensions());
        assertEquals(original.getActivationFunctionName(), copy.getActivationFunctionName());
    }

    /**
     * Testa o método `merge` para verificar a combinação de duas redes neurais.
     */
    @Test
    public void testMerge() {
        NeuralNetwork nn1 = new NeuralNetwork(4, 1, 5, 1);
        NeuralNetwork nn2 = new NeuralNetwork(4, 1, 5, 1);

        NeuralNetwork merged = nn1.merge(nn2, 0.5);

        assertNotNull(merged);
        assertArrayEquals(nn1.getDimensions(), merged.getDimensions());
    }

    /**
     * Testa se o método `mutate` altera os pesos e vieses corretamente.
     */
    @Test
    public void testMutate() {
        NeuralNetwork nn = new NeuralNetwork(4, 1, 5, 1);
        nn.setActivationFunction("SIGMOID");

        SimpleMatrix[] initialWeights = new SimpleMatrix[nn.getWeights().length];
        for (int i = 0; i < nn.getWeights().length; i++) {
            initialWeights[i] = nn.getWeights()[i].copy();
        }

        SimpleMatrix[] initialBiases = new SimpleMatrix[nn.getBiases().length];
        for (int i = 0; i < nn.getBiases().length; i++) {
            initialBiases[i] = nn.getBiases()[i].copy();
        }

        nn.mutate(0.5);

        boolean weightsChanged = false;
        for (int i = 0; i < initialWeights.length; i++) {
            if (!initialWeights[i].isIdentical(nn.getWeights()[i], 1e-6)) {
                weightsChanged = true;
                break;
            }
        }

        boolean biasesChanged = false;
        for (int i = 0; i < initialBiases.length; i++) {
            if (!initialBiases[i].isIdentical(nn.getBiases()[i], 1e-6)) {
                biasesChanged = true;
                break;
            }
        }

        assertTrue("Pesos ou vieses deveriam mudar após a mutação", weightsChanged || biasesChanged);
    }


    /**
     * Testa a propagação direta com entradas conhecidas.
     */
    @Test
    public void testForwardPropagation() {
        NeuralNetwork nn = new NeuralNetwork(4, 1, 5, 1);
        nn.setActivationFunction("SIGMOID");

        double[] input = {0.1, 0.2, 0.3, 0.4};
        double[] output = nn.guess(input);

        assertNotNull(output);
        assertEquals(1, output.length);
        assertTrue(output[0] >= 0 && output[0] <= 1); // Saída sigmoidal
    }
}
