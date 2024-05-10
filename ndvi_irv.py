#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
file_path = r'C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\MNDWI_NDVI_IRV.csv'
data = pd.read_csv(file_path)

ce_df = pd.read_csv('C://Users//camil//Downloads//Salinidad//Primera campaña SALINIDAD//Campaña_abril2023//suelo_abr2023.csv', sep=';', decimal='.')
CE = ce_df[['CE']]

data = pd.concat([data, CE], axis=1)

#%% IRV vs NDVI
correlacion_IRV_NDVI = data['IRV'].corr(data['NDVI'])
print(f"La correlación de Pearson entre 'IRV' y 'NDVI' es: {correlacion_IRV_NDVI}")

pendiente1 = correlacion_IRV_NDVI * (data['IRV'].std() / data['NDVI'].std())
interc1 = data['IRV'].mean() - pendiente1 * data['NDVI'].mean()
x_val1 = np.linspace(0, 1)
y_val1 = pendiente1 * x_val1 + interc1
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data['NDVI'], data['IRV'], c=data['CE'], cmap='viridis')
plt.colorbar(scatter, label='CE') 
plt.xlabel('NDVI')
plt.ylabel('IRV')
plt.xlim(0, 0.43)
plt.ylim(0, 1.1)
plt.title("IRV (SAOCOM) VS. NDVI (Sentinel-2)")
plt.grid(True)
plt.gca().set_facecolor('#f0f0f0') 
plt.plot(x_val1, y_val1, color='red', linestyle='--', label=f'Correlación = {correlacion_IRV_NDVI:.2f}')
plt.legend()
plt.show()

#%% IRV vs MNDWI
correlacion_IRV_MNDWI = data['IRV'].corr(data['MNDWI'])
print(f"La correlación de Pearson entre 'IRV' y 'MNDWI' es: {correlacion_IRV_MNDWI}")

covarianza = data['MNDWI'].cov(data['IRV'])
varianza_mndwi = data['MNDWI'].var()
mean_irv = data['IRV'].mean()
mean_mndwi = data['MNDWI'].mean()
pendiente2 = covarianza / varianza_mndwi
interc2 = mean_irv - pendiente2 * mean_mndwi
x_val2 = np.linspace(-0.25, 0, 100)
y_val2 = pendiente2 * x_val2 + interc2

plt.figure(figsize=(10, 6))
scatter = plt.scatter(data['MNDWI'], data['IRV'], c=data['CE'], cmap='viridis')
plt.colorbar(scatter, label='CE') 
plt.xlabel('MNDWI')
plt.ylabel('IRV')
plt.xlim(-0.25, 0)
plt.ylim(0, 0.85)
plt.title("IRV (SAOCOM) VS. MNDWI (Sentinel-2)")
plt.grid(True)
plt.gca().set_facecolor('#f0f0f0') 
plt.plot(x_val2, y_val2, color='red', linestyle='--', label=f'Correlación = {correlacion_IRV_MNDWI:.2f}')
plt.legend()
plt.show()









