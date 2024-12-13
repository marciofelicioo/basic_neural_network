package basicneuralnetwork.neuralnetwork;

import java.io.BufferedReader;
import java.io.InputStreamReader;
/**
 * Classe DigitClassifier: Implementa um programa para classificar dígitos usando uma rede neural pré-treinada.
 * O programa lê uma linha de entrada contendo 400 valores de pixels, carrega o modelo treinado, realiza a predição e imprime o resultado (0 ou 1).
 *
 * @author Márcio Felício, Maria Anjos, Miguel Rosa
 * @version 1.0 30/11/2024
 *
 * @inv
 * - A entrada deve conter exatamente 400 valores numéricos separados por vírgula, representando os pixels de uma imagem 20x20.
 * - Os valores de entrada devem ser normalizados da mesma forma que durante o treinamento (divididos por 255.0).
 * - O modelo da rede neural deve ser carregado com sucesso a partir do arquivo especificado (`model_weights.txt`).
 * - A predição realizada pela rede neural deve resultar em um valor no intervalo [0, 1].
 * - O programa deve interpretar corretamente a saída da rede neural e imprimir apenas `0` ou `1`.
 * - Em caso de erro na leitura da entrada ou carregamento dos pesos, o programa deve lidar com a exceção de forma apropriada.
 */
public class DigitClassifier {
    public static void main(String[] args) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String inputLine = reader.readLine();
            String[] inputValues = inputLine.trim().split(",");

            if (inputValues.length != 400) {
                throw new IllegalArgumentException("Erro: Esperados 400 valores de entrada.");
            }

            for (String value : inputValues) {
                try {
                    Double.parseDouble(value);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("Erro: Valor inválido encontrado na entrada: " + value);
                }
            }


            double[] inputPixels = new double[400];
            for (int i = 0; i < 400; i++) {
                inputPixels[i] = Double.parseDouble(inputValues[i]);
                inputPixels[i] /= 255.0;
            }

            NeuralNetwork nn = new NeuralNetwork(400, 10, 1);
            nn.setActivationFunction("SIGMOID");

            try {
                nn.loadWeights("src/main/java/model_weights.txt");
            } catch (Exception e) {
                System.err.println("Erro ao carregar os pesos: " + e.getMessage());
                return;
            }

            double[] output = nn.guess(inputPixels);

            int prediction = output[0] >= 0.5 ? 1 : 0;

            System.out.println(prediction);
        } catch (Exception e) {
            System.err.println("Erro ao processar a entrada: " + e.getMessage());
        }
    }
}
