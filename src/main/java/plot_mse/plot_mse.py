import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("mse_history.csv")


plt.figure(figsize=(10, 6))
plt.plot(data['Iteração'], data['MSE_Treino'], label='MSE Treino')
plt.plot(data['Iteração'], data['MSE_Validação'], label='MSE Validação', linestyle='--')


plt.title('Evolução do MSE durante o Treinamento')
plt.xlabel('Iterações')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)


plt.savefig("mse_plot.png")
plt.show()
