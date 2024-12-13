package basicneuralnetwork.neuralnetwork;

import java.io.*;
import java.util.*;
/**
 * Classe DataPreprocessor: Responsável pelo pré-processamento do conjunto de dados,
 * incluindo carregamento, validação, normalização e divisão em conjuntos de treinamento e teste.
 *
 * @author Márcio Felício, Maria Anjos, Miguel Rosa
 * @version 1.0 30/11/2024
 *
 * @inv
 * - Cada linha do arquivo `dataset.csv` deve conter exatamente 400 valores, representando os pixels de uma imagem 20x20.
 * - Cada linha do arquivo `labels.csv` deve conter apenas os valores 0 ou 1, representando os rótulos válidos.
 * - O número de linhas nos arquivos `dataset.csv` e `labels.csv` deve ser igual.
 * - Durante a normalização, todos os valores de pixel devem ser convertidos para o intervalo [0, 1].
 * - O conjunto de dados deve ser corretamente dividido em treinamento e teste com base na proporção especificada.
 */
public class DataPreprocessor {

    /**
     * Carrega o conjunto de dados e os rótulos de arquivos CSV.
     * Realiza a validação, normalização dos dados e adiciona o rótulo como último valor de cada linha.
     *
     * @param dataFilePath Caminho para o arquivo `dataset.csv` contendo os valores dos pixels.
     * @param labelFilePath Caminho para o arquivo `labels.csv` contendo os rótulos.
     * @return Uma lista contendo arrays de double, onde cada array representa uma imagem normalizada e seu rótulo.
     * @throws IOException Se houver inconsistências nos arquivos ou erros de leitura.
     */
    public static List<double[]> loadDataset(String dataFilePath, String labelFilePath) throws IOException {
        List<double[]> dataset = new ArrayList<>();
        BufferedReader dataReader = new BufferedReader(new FileReader(dataFilePath));
        BufferedReader labelReader = new BufferedReader(new FileReader(labelFilePath));

        String dataLine;
        int lineCount = 0;

        while ((dataLine = dataReader.readLine()) != null) {
            String labelLine = labelReader.readLine();

            if (labelLine == null) {
                throw new IOException("Erro: `labels.csv` tem menos linhas do que `dataset.csv`.");
            }

            String[] pixelValues = dataLine.split(",");
            if (pixelValues.length != 400) {
                throw new IOException(String.format("Erro na linha %d: `dataset.csv` tem %d valores (esperado: 400).", lineCount + 1, pixelValues.length));
            }

            int label;
            try {
                label = Integer.parseInt(labelLine.trim());
                if (label != 0 && label != 1) {
                    throw new IOException(String.format("Erro na linha %d: Rótulo inválido em `labels.csv` (%d).", lineCount + 1, label));
                }
            } catch (NumberFormatException e) {
                throw new IOException(String.format("Erro na linha %d: Valor inválido em `labels.csv` (%s).", lineCount + 1, labelLine));
            }

            double[] input = new double[pixelValues.length];
            for (int i = 0; i < pixelValues.length; i++) {
                double value = Double.parseDouble(pixelValues[i]);
                input[i] = Math.max(0, value) / 255.0;
            }

            double[] row = Arrays.copyOf(input, input.length + 1);
            row[input.length] = label;

            dataset.add(row);
            lineCount++;
        }

        if (labelReader.readLine() != null) {
            throw new IOException("Erro: `labels.csv` tem mais linhas do que `dataset.csv`.");
        }

        dataReader.close();
        labelReader.close();

        return dataset;
    }

    /**
     * Divide o conjunto de dados em treinamento e teste com base na proporção especificada.
     *
     * @param dataset Conjunto de dados a ser dividido.
     * @param splitRatio Proporção de divisão (ex: 0.8 para 80% de treinamento e 20% de teste).
     * @param trainSet Lista onde os dados de treinamento serão armazenados.
     * @param testSet Lista onde os dados de teste serão armazenados.
     */
    public static void splitDataset(List<double[]> dataset, double splitRatio, List<double[]> trainSet, List<double[]> testSet) {
        int splitIndex = (int) (dataset.size() * splitRatio);
        trainSet.addAll(dataset.subList(0, splitIndex));
        testSet.addAll(dataset.subList(splitIndex, dataset.size()));
    }
}
