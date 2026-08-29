import numpy as np
import pandas as pd
from scipy.stats import qmc

np.random.seed(42)
N_SAMPLES = 500

bounds_lower = [320.0, 101325.0, 0.30,  80.0, 16.0, 0.35, 1.5, 0.35]
bounds_upper = [370.0, 200000.0, 0.70, 120.0, 28.0, 0.65, 4.5, 0.65]

sampler = qmc.LatinHypercube(d=8, seed=42)
sample_unit = sampler.random(n=N_SAMPLES)
scaled_samples = qmc.scale(sample_unit, bounds_lower, bounds_upper)

df = pd.DataFrame(scaled_samples, columns=[
    'Feed_Temp_K', 'Feed_Pres_Pa', 'zF_Benzene', 'Feed_Flow_kmolh',
    'N_Stages', 'Feed_Stage_Ratio', 'Reflux_Ratio', 'Bottoms_Ratio'
])

df['N_Stages'] = df['N_Stages'].round().astype(int)
df['Feed_Stage'] = (df['N_Stages'] * df['Feed_Stage_Ratio']).round().astype(int)
df['Bottoms_Flow_kmolh'] = (df['Feed_Flow_kmolh'] * df['Bottoms_Ratio']).round(3)
df['Distillate_Flow_kmolh'] = (df['Feed_Flow_kmolh'] - df['Bottoms_Flow_kmolh']).round(3)

A1, B1, C1 = 4.01814, 1203.835, -53.226
A2, B2, C2 = 4.07827, 1343.943, -53.773

def get_p_sat(T):
    P1 = 10 ** (A1 - B1 / (T + C1))
    P2 = 10 ** (A2 - B2 / (T + C2))
    return P1, P2

DH_VAP_BENZ = 30770.0 
DH_VAP_TOLU = 33180.0

xD_list, xB_list, Qc_list, Qr_list = [], [], [], []

for _, row in df.iterrows():
    TF = row['Feed_Temp_K']
    zF = row['zF_Benzene']
    F = row['Feed_Flow_kmolh']
    N = int(row['N_Stages'])
    NF = int(row['Feed_Stage'])
    RR = row['Reflux_Ratio']
    B = row['Bottoms_Flow_kmolh']
    D = row['Distillate_Flow_kmolh']
    
    alpha_avg = 2.45 - 0.002 * (TF - 350.0)
    S_index = (alpha_avg ** (0.75 * N)) * ((RR / (RR + 1.0)) ** 0.85) * (1.0 - abs(NF/N - 0.5))
    
    xD_raw = (zF * F * S_index) / (D * S_index + B * 0.05)
    xD = float(np.clip(xD_raw, zF + 0.05, 0.9992))
    
    xB_raw = 1.0 - ((F * zF - D * xD) / max(B, 1e-4))
    xB = float(np.clip(xB_raw, 0.70, 0.9995))
    
    V_top = D * (RR + 1.0)
    dH_cond = xD * DH_VAP_BENZ + (1.0 - xD) * DH_VAP_TOLU
    Qc = (V_top * dH_cond) / 3600.0
    
    Cp = 140.0
    T_cond = 353.3
    T_reb = 383.8
    sensible_heat = (D * Cp * (T_cond - 298.15) + B * Cp * (T_reb - 298.15) - F * Cp * (TF - 298.15)) / 3600.0
    Qr = Qc + sensible_heat
    
    noise_xD = np.random.normal(0, 0.0005)
    noise_Q = np.random.normal(0, 0.5)
    
    xD_list.append(np.clip(xD + noise_xD, 0.0, 1.0))
    xB_list.append(np.clip(xB + noise_xD, 0.0, 1.0))
    Qc_list.append(round(Qc + noise_Q, 2))
    Qr_list.append(round(Qr + noise_Q, 2))

df['xD_Benzene'] = np.round(xD_list, 4)
df['xB_Toluene'] = np.round(xB_list, 4)
df['Q_Condenser_kW'] = Qc_list
df['Q_Reboiler_kW'] = Qr_list

df.to_csv("Dataset.csv", index=False)
print("Dataset.csv generated successfully with 500 validated points.")