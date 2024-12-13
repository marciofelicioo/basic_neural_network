package UnitTests;

import basicneuralnetwork.neuralnetwork.DataPreprocessor;
import org.junit.Test;

import java.io.*;
import java.util.*;

import static org.junit.Assert.*;

/**
 * Classe DataPreprocessorTests: Testes unitários para os métodos da classe DataPreprocessor.
 *
 * @author Márcio Felício, Maria Anjos, Miguel Rosa
 * @version 1.0 30/11/2024
 */
public class DataPreprocessorTests {

    /**
     * Testa o método `loadDataset` para verificar se os dados são carregados corretamente.
     */
    @Test
    public void testLoadDatasetValidFiles() throws IOException {
        String dataFilePath = "dataset/dataset/dataset.csv";
        String labelFilePath = "dataset/dataset/labels.csv";

        List<double[]> dataset = DataPreprocessor.loadDataset(dataFilePath, labelFilePath);

        assertEquals(800, dataset.size());
        for (double[] row : dataset) {
            assertEquals(401, row.length); // 400 pixels + 1 rótulo
        }

        for (double[] row : dataset) {
            for (int i = 0; i < 400; i++) {
                assertTrue(row[i] >= 0 && row[i] <= 1);
            }
        }

        for (double[] row : dataset) {
            assertTrue(row[400] == 0 || row[400] == 1);
        }
    }

    /**
     * Testa o método `loadDataset` com arquivos mal formatados.
     */
    @Test(expected = IOException.class)
    public void testLoadDatasetMismatchedLines() throws IOException {
        String dataFilePath = "src/test/resources/dataset_mismatch.csv";
        String labelFilePath = "src/test/resources/labels.csv";

        DataPreprocessor.loadDataset(dataFilePath, labelFilePath);
    }

    @Test(expected = IOException.class)
    public void testLoadDatasetInvalidPixelCount() throws IOException {
        String dataFilePath = "src/test/resources/dataset_invalid_pixels.csv";
        String labelFilePath = "src/test/resources/labels.csv";

        DataPreprocessor.loadDataset(dataFilePath, labelFilePath);
    }

    @Test(expected = IOException.class)
    public void testLoadDatasetInvalidLabels() throws IOException {
        String dataFilePath = "src/test/resources/dataset.csv";
        String labelFilePath = "src/test/resources/labels_invalid.csv";

        DataPreprocessor.loadDataset(dataFilePath, labelFilePath);
    }

    /**
     * Testa o método `splitDataset` para verificar se o dataset é dividido corretamente.
     */
    @Test
    public void testSplitDataset() {
        List<double[]> dataset = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            dataset.add(new double[401]); // 400 pixels + 1 rótulo
        }

        List<double[]> trainSet = new ArrayList<>();
        List<double[]> testSet = new ArrayList<>();

        DataPreprocessor.splitDataset(dataset, 0.8, trainSet, testSet);


        assertEquals(80, trainSet.size());
        assertEquals(20, testSet.size());
    }

    /**
     * Testa o método `splitDataset` para verificar comportamento com proporções extremas.
     */
    @Test
    public void testSplitDatasetExtremeRatios() {
        List<double[]> dataset = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            dataset.add(new double[401]);
        }

        List<double[]> trainSet = new ArrayList<>();
        List<double[]> testSet = new ArrayList<>();
        DataPreprocessor.splitDataset(dataset, 1.0, trainSet, testSet);
        assertEquals(10, trainSet.size());
        assertEquals(0, testSet.size());

        trainSet.clear();
        testSet.clear();
        DataPreprocessor.splitDataset(dataset, 0.0, trainSet, testSet);
        assertEquals(0, trainSet.size());
        assertEquals(10, testSet.size());
    }
}
