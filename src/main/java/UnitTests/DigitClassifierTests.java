package UnitTests;

import basicneuralnetwork.neuralnetwork.DigitClassifier;
import org.junit.Test;

import java.io.*;

import static org.junit.Assert.*;

/**
 * Classe DigitClassifierTests: Testes unitários para a classe DigitClassifier.
 * Verifica a funcionalidade de classificação de dígitos usando uma rede neural pré-treinada.
 *
 * @author Márcio Felício, Maria Anjos, Miguel Rosa
 * @version 1.0 30/11/2024
 */
public class DigitClassifierTests {

    /**
     * Método auxiliar para gerar uma string com valores repetidos.
     */
    private String generateRepeatedString(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }

    /**
     * Testa a classificação de entrada válida.
     */
    @Test
    public void testValidInput() {
        String input = generateRepeatedString("0,", 399) + "0";

        InputStream originalIn = System.in;
        PrintStream originalOut = System.out;
        PrintStream originalErr = System.err;

        ByteArrayInputStream testInput = new ByteArrayInputStream(input.getBytes());
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        ByteArrayOutputStream errorStream = new ByteArrayOutputStream();

        try {
            System.setIn(testInput);
            System.setOut(new PrintStream(outputStream));
            System.setErr(new PrintStream(errorStream));

            DigitClassifier.main(new String[]{});

            String output = outputStream.toString().trim();
            String errorOutput = errorStream.toString().trim();

            assertTrue(errorOutput.isEmpty());

            assertTrue(output.equals("0") || output.equals("1"));
        } finally {
            System.setIn(originalIn);
            System.setOut(originalOut);
            System.setErr(originalErr);
        }
    }

    /**
     * Testa entrada com número incorreto de pixels.
     */
    @Test
    public void testInvalidPixelCount() {
        String input = generateRepeatedString("0,", 19) + "0";

        InputStream originalIn = System.in;
        PrintStream originalErr = System.err;

        ByteArrayInputStream testInput = new ByteArrayInputStream(input.getBytes());
        ByteArrayOutputStream errorStream = new ByteArrayOutputStream();

        try {
            System.setIn(testInput);
            System.setErr(new PrintStream(errorStream));

            DigitClassifier.main(new String[]{});

            String errorOutput = errorStream.toString().trim();

            assertTrue(errorOutput.contains("Erro ao processar a entrada: Erro: Esperados 400 valores de entrada."));
        } finally {
            System.setIn(originalIn);
            System.setErr(originalErr);
        }
    }

    /**
     * Testa entrada com valores inválidos.
     */
    @Test
    public void testInvalidPixelValues() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 199; i++) {
            sb.append("0,");
        }
        sb.append("a,");
        for (int i = 0; i < 200; i++) {
            sb.append("0,");
        }
        sb.deleteCharAt(sb.length() - 1);
        String input = sb.toString();

        InputStream originalIn = System.in;
        PrintStream originalErr = System.err;

        ByteArrayInputStream testInput = new ByteArrayInputStream(input.getBytes());
        ByteArrayOutputStream errorStream = new ByteArrayOutputStream();

        try {
            System.setIn(testInput);
            System.setErr(new PrintStream(errorStream));

            DigitClassifier.main(new String[]{});

            String errorOutput = errorStream.toString().trim();

            assertTrue(errorOutput.contains("Erro ao processar a entrada: Erro: Valor inválido encontrado na entrada: a"));
        } finally {
            System.setIn(originalIn);
            System.setErr(originalErr);
        }
    }

    /**
     * Testa o comportamento ao falhar ao carregar os pesos.
     */
    @Test
    public void testWeightLoadFailure() {
        String input = generateRepeatedString("0,", 399) + "0";

        InputStream originalIn = System.in;
        PrintStream originalErr = System.err;

        ByteArrayInputStream testInput = new ByteArrayInputStream(input.getBytes());
        ByteArrayOutputStream errorStream = new ByteArrayOutputStream();

        File weightsFile = new File("src/main/java/model_weights.txt");
        File tempFile = new File("model_weights_backup.txt");
        boolean renamed = weightsFile.renameTo(tempFile);

        try {
            System.setIn(testInput);
            System.setErr(new PrintStream(errorStream));

            DigitClassifier.main(new String[]{});

            String errorOutput = errorStream.toString().trim();

            assertTrue(errorOutput.contains("Erro ao carregar os pesos:"));

        } finally {
            if (renamed) {
                tempFile.renameTo(weightsFile);
            }
            System.setIn(originalIn);
            System.setErr(originalErr);
        }
    }
}
