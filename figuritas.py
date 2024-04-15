import numpy as np
import pandas as pd


#creamos un dominio de 10 matrices de 0s de 10x10
domain = np.zeros((10,10, 10))
print(domain)

df = pd.DataFrame({'x':[0,1], 'y':[0,1], 'z':[0,1], 'pressure':[100,150]})
df.to_csv('data.csv', index=False)
