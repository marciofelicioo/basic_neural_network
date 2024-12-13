package UnitTests;

import basicneuralnetwork.neuralnetwork.DigitTrainer;
import basicneuralnetwork.neuralnetwork.NeuralNetwork;
import basicneuralnetwork.neuralnetwork.DataPreprocessor;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Classe DigitTrainerTests: Testes unitários para a classe DigitRecognizer e métodos relacionados.
 *
 * @author Márcio Felício, Maria Anjos, Miguel Rosa
 * @version 1.0 30/11/2024
 */
public class DigitTrainerTests {

    /**
     * Testa o método de carregamento de dados para verificar se os dados são carregados corretamente.
     */
    @Test
    public void testLoadDataset() {
        String dataFilePath = "dataset/dataset/dataset.csv";
        String labelFilePath = "dataset/dataset/labels.csv";

        try {
            List<double[]> dataset = DataPreprocessor.loadDataset(dataFilePath, labelFilePath);

            assertEquals(800, dataset.size());
            for (double[] row : dataset) {
                assertEquals(401, row.length);
            }

        } catch (Exception e) {
            fail("Erro ao carregar o dataset: " + e.getMessage());
        }
    }

    /**
     * Testa o método de divisão de dataset para verificar se os conjuntos de treinamento e teste estão corretos.
     */
    @Test
    public void testSplitDataset() {
        List<double[]> dataset = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            dataset.add(new double[401]);
        }

        List<double[]> trainSet = new ArrayList<>();
        List<double[]> testSet = new ArrayList<>();

        DataPreprocessor.splitDataset(dataset, 0.8, trainSet, testSet);

        assertEquals(80, trainSet.size());
        assertEquals(20, testSet.size());
    }

    /**
     * Testa o cálculo do MSE para verificar se o erro é calculado corretamente.
     */
    @Test
    public void testCalculateMSE() {
        NeuralNetwork nn = new NeuralNetwork(400, 10, 1);
        nn.setActivationFunction("SIGMOID");

        List<double[]> dataset = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            double[] row = new double[401];
            for (int j = 0; j < 400; j++) {
                row[j] = i * 0.1;
            }
            row[400] = (i % 2 == 0) ? 0 : 1;
            dataset.add(row);
        }

        double mse = DigitTrainer.calculateMSE(nn, dataset);
        assertTrue(mse >= 0);
    }

    /**
     * Testa o treinamento da rede neural com um conjunto simples de dados.
     */
    @Test
    public void testTrainNeuralNetwork() {
        NeuralNetwork nn = new NeuralNetwork(400, 10, 1);
        nn.setActivationFunction("SIGMOID");
        nn.setLearningRate(0.01);

        List<double[]> trainSet = new ArrayList<>();
        List<double[]> validationSet = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            double[] row = new double[401];
            for (int j = 0; j < 400; j++) {
                row[j] = Math.random();
            }
            row[400] = (i % 2 == 0) ? 0 : 1;
            trainSet.add(row);
            validationSet.add(row);
        }

        DigitTrainer.trainNeuralNetwork(nn, trainSet, validationSet, 0.001, 100, 5);
    }

    /**
     * Testa a avaliação da rede neural para verificar a precisão no conjunto de teste.
     */
    @Test
    public void testEvaluateNeuralNetwork() {
        NeuralNetwork nn = new NeuralNetwork(400, 10, 1);
        nn.setActivationFunction("SIGMOID");

        List<double[]> testSet = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            double[] row = new double[401];
            for (int j = 0; j < 400; j++) {
                row[j] = Math.random();
            }
            row[400] = (i % 2 == 0) ? 0 : 1;
            testSet.add(row);
        }

        DigitTrainer.evaluateNeuralNetwork(nn, testSet);
    }
}
