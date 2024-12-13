package basicneuralnetwork.neuralnetwork;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
/**
 * Classe DigitTrainer: Implementa um programa de reconhecimento de dígitos usando redes neurais.
 * A classe gerencia o carregamento, pré-processamento de dados, treinamento da rede neural e avaliação.
 *
 * @author Márcio Felício, Maria Anjos, Miguel Rosa
 * @version 1.0 30/11/2024
 *
 * @inv
 * - Os dados carregados (dataset) devem conter exatamente 400 valores por linha, representando os pixels de 20x20 de uma imagem.
 * - Cada rótulo no arquivo de rótulos deve ser 0 ou 1, garantindo que o modelo seja treinado apenas com esses dois dígitos.
 * - Durante o treinamento:
 *   - O erro médio quadrático (MSE) deve diminuir ou estabilizar a cada iteração.
 *   - O número de iterações não deve exceder o limite máximo definido.
 * - O conjunto de dados deve ser corretamente dividido em treinamento (80%) e teste (20%).
 * - A rede neural deve retornar previsões no intervalo [0, 1] devido ao uso da função de ativação sigmoide.
 */
public class DigitTrainer {
    /**
     * Lista para armazenar o histórico de MSE para treinamento e validação
     */
    private static final List<Double> trainMseHistory = new ArrayList<>();
    private static final List<Double> validationMseHistory = new ArrayList<>();

    /**
     * Exporta o histórico do MSE para um arquivo CSV para posterior visualização.
     *
     * @param trainMseHistory Histórico do MSE do conjunto de treinamento.
     * @param validationMseHistory Histórico do MSE do conjunto de validação.
     * @param filePath Caminho para salvar o arquivo CSV.
     * @throws IOException Em caso de erro ao escrever o arquivo.
     */
    public static void exportMseHistoryToCsv(List<Double> trainMseHistory, List<Double> validationMseHistory, String filePath) throws IOException {
        try (FileWriter writer = new FileWriter(filePath)) {
            writer.write("Iteração,MSE_Treino,MSE_Validação\n");

            for (int i = 0; i < trainMseHistory.size(); i++) {
                writer.write((i + 1) + "," + trainMseHistory.get(i) + "," + validationMseHistory.get(i) + "\n");
            }
        }
    }

    /**
     * Função principal que inicializa a execução do programa.
     * Configura a rede neural, realiza o pré-processamento dos dados,
     * treina a rede neural, avalia o desempenho e calcula o tempo total de execução.
     *
     * @param args Argumentos da linha de comando.
     * @throws IOException Em caso de erro ao carregar os arquivos do conjunto de dados.
     */
    public static void main(String[] args) throws IOException {
        long startTime = System.nanoTime();


        NeuralNetwork nn = new NeuralNetwork(400, 10, 1);
        nn.setActivationFunction("SIGMOID");
        nn.setLearningRate(0.01);


        String dataFilePath = "dataset/dataset/dataset.csv";
        String labelFilePath = "dataset/dataset/labels.csv";
        List<double[]> dataset = DataPreprocessor.loadDataset(dataFilePath, labelFilePath);


        List<double[]> trainSet = new ArrayList<>();
        List<double[]> testSet = new ArrayList<>();
        DataPreprocessor.splitDataset(dataset, 0.6, trainSet, testSet);


        trainNeuralNetwork(nn, trainSet, testSet, 0.001, 2000, 10);

        exportMseHistoryToCsv(trainMseHistory, validationMseHistory, "src/main/java/plot_mse/mse_history.csv");

        evaluateNeuralNetwork(nn, testSet);

        try {
            nn.saveWeights("src/main/java/model_weights.txt");
            System.out.println("Pesos da rede neural salvos com sucesso.");
        } catch (IOException e) {
            System.err.println("Erro ao salvar os pesos: " + e.getMessage());
        }
        long endTime = System.nanoTime();
        double totalTime = (endTime - startTime) / 1e9;
        System.out.printf("\nTempo total de execução: %.3f segundos%n", totalTime);
    }

    /**
     * Calcula o erro médio quadrático (MSE) da rede neural para um conjunto de dados.
     *
     * @param nn A instância da rede neural.
     * @param dataset O conjunto de dados a ser avaliado.
     * @return O valor do MSE calculado.
     */
    public static double calculateMSE(NeuralNetwork nn, List<double[]> dataset) {
        double mse = 0.0;

        for (double[] row : dataset) {
            double[] input = Arrays.copyOfRange(row, 0, 400);
            double target = row[400];
            double prediction = nn.guess(input)[0];
            mse += Math.pow(prediction - target, 2);
        }

        return mse / dataset.size();
    }

    /**
     * Treina a rede neural com divisão do conjunto de dados em treinamento e validação.
     * Utiliza Early Stopping para evitar overfitting.
     *
     * @param nn A instância da rede neural.
     * @param trainSet Conjunto de treinamento.
     * @param validationSet Conjunto de validação.
     * @param mseThreshold Limiar para o MSE.
     * @param maxIterations Número máximo de iterações.
     * @param patience Número de iterações sem melhora antes de parar.
     */
    public static void trainNeuralNetwork(NeuralNetwork nn, List<double[]> trainSet, List<double[]> validationSet, double mseThreshold, int maxIterations, int patience) {
        double bestValidationMSE = Double.MAX_VALUE;
        int patienceCounter = 0;

        System.out.println("\nInício do treinamento da rede neural com Early Stopping...\n");

        for (int iteration = 1; iteration <= maxIterations; iteration++) {
            double trainMSE = 0.0;

            for (double[] row : trainSet) {
                double[] input = Arrays.copyOfRange(row, 0, 400);
                double[] target = {row[400]};
                nn.train(input, target);

                double prediction = nn.guess(input)[0];
                trainMSE += Math.pow(prediction - target[0], 2);
            }
            trainMSE /= trainSet.size();

            double validationMSE = calculateMSE(nn, validationSet);

            trainMseHistory.add(trainMSE);
            validationMseHistory.add(validationMSE);

            System.out.printf("Iteração %d - MSE Treino: %.5f, MSE Validação: %.5f%n", iteration, trainMSE, validationMSE);

            if (validationMSE < bestValidationMSE) {
                bestValidationMSE = validationMSE;
                patienceCounter = 0;
            } else {
                patienceCounter++;
            }

            if (validationMSE <= mseThreshold) {
                System.out.printf("Parada antecipada: Erro no conjunto de validação atingiu o limiar (%.5f).%n", mseThreshold);
                break;
            }
            if (patienceCounter >= patience) {
                System.out.printf("Parada antecipada: Nenhuma melhora no conjunto de validação por %d iterações.%n", patience);
                break;
            }
        }

        System.out.println("\nTreinamento concluído.");
    }


    /**
     * Avalia o desempenho da rede neural no conjunto de teste.
     * Exibe a acurácia, previsões e erros no console.
     *
     * @param nn A instância da rede neural.
     * @param testSet O conjunto de teste.
     */
    public static void evaluateNeuralNetwork(NeuralNetwork nn, List<double[]> testSet) {
        int correct = 0;
        int totalSamples = testSet.size();

        System.out.println("\nAvaliação no conjunto de teste:");
        for (double[] row : testSet) {
            double[] input = Arrays.copyOfRange(row, 0, 400);
            double target = row[400];
            double prediction = nn.guess(input)[0];
            int predictedLabel = (prediction >= 0.5) ? 1 : 0;

            System.out.printf("Rótulo Real: %.1f, Previsão (Sigmoid): %.5f, Rótulo Previsto: %d%n",
                    target, prediction, predictedLabel);

            if (predictedLabel == (int) target) {
                correct++;
            } else {
                System.out.printf("Erro - Previsto: %d, Real: %d%n", predictedLabel, (int) target);
            }
        }

        double accuracy = (correct / (double) totalSamples) * 100;
        System.out.println("\nResultados finais:");
        System.out.printf("Acurácia no teste: %.2f%%%n", accuracy);
        System.out.printf("Amostras corretas: %d / %d%n", correct, totalSamples);
    }
}
