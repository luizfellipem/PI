import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar o estilo dos gráficos
sns.set(style="whitegrid")

# Gerar dados fictícios
np.random.seed(42)

# Número de crianças analisadas
n = 100

# Dados simulados
data = {
    'Distancia_hospital_km': np.random.normal(50, 20, n),  # média de 50km, desvio padrão de 20km
    'Tempo_espera_transporte_dias': np.random.randint(0, 10, n),  # dias de espera pelo transporte
    'Frequencia_faltas_%': np.random.normal(15, 10, n),  # % de faltas, média de 15%
    'Impacto_na_saude': np.random.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1]),  # 0 = sem impacto, 1 = moderado, 2 = grave
    'Dificuldade_financeira_%': np.random.randint(40, 100, n),  # % de famílias com dificuldades financeiras
    'Numero_hospitais_proximos': np.random.randint(0, 3, n)  # Quantidade de hospitais próximos
}

# Criar DataFrame
df = pd.DataFrame(data)

# Limitar os dados para valores realistas
df['Frequencia_faltas_%'] = df['Frequencia_faltas_%'].clip(lower=0, upper=100)
df['Distancia_hospital_km'] = df['Distancia_hospital_km'].clip(lower=0)

# Análise descritiva
print(df.describe())

plt.figure(figsize=(8, 6))
sns.histplot(df['Distancia_hospital_km'], bins=20, kde=True)
plt.title('Distribuição da Distância até o Hospital')
plt.xlabel('Distância (km)')
plt.ylabel('Frequência')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Impacto_na_saude', y='Frequencia_faltas_%', data=df)
plt.title('Impacto na Saúde vs. Faltas ao Tratamento')
plt.xlabel('Impacto na Saúde (0=Sem, 1=Moderado, 2=Grave)')
plt.ylabel('% de Faltas')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['Tempo_espera_transporte_dias'], bins=10, kde=False)
plt.title('Distribuição do Tempo de Espera por Transporte')
plt.xlabel('Tempo de Espera (dias)')
plt.ylabel('Frequência')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Distancia_hospital_km', y='Dificuldade_financeira_%', data=df)
plt.title('Correlação entre Distância ao Hospital e Dificuldade Financeira')
plt.xlabel('Distância (km)')
plt.ylabel('% de Dificuldade Financeira')
plt.show()

import statsmodels.api as sm

# Variáveis dependente e independente
X = df['Distancia_hospital_km']
y = df['Frequencia_faltas_%']

# Adicionar constante para o modelo
X = sm.add_constant(X)

# Ajustar o modelo de regressão linear
model = sm.OLS(y, X).fit()

# Exibir o resumo do modelo
print(model.summary())
