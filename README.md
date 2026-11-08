<div style="width:100%; background-color:#00522c; border-bottom:3px solid #34a853; padding:20px 0; text-align:center;">
  <img src="https://pattern-lab-externado-prod.web.app/images/logo-uec.svg" alt="Banner del proyecto" width="40%">
</div>

# Hacia un sistema de valuación masiva con BART para la vivienda de Bogotá y alrededores


![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style=for-the-badge&logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?style=for-the-badge&logo=scikitlearn&logoColor=white)
![stochtree](https://img.shields.io/badge/stochtree-2E7D32.svg?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD.svg?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B.svg?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-FF0D57.svg?style=for-the-badge)
![ArviZ](https://img.shields.io/badge/ArviZ-0F4C81.svg?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C.svg?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0.svg?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## Abstract
In Colombia, access to official data on actual housing sale prices is limited, and when available, it is usually unreliable due to phenomena such as tax underreporting. Within this framework, a prototype Mass Appraisal System (MAS) is proposed for used housing in Bogotá, Soacha, Chía and Zipaquirá, using asking prices collected through *web scraping* from the FincaRaíz portal. The main model is BART(Atr), an adaptation of BART with adaptive outlier trimming proposed by Cao et al. (2024), which is compared through cross-validation against OLS, Random Forest, XGBoost, LightGBM and a median predictor. On the test set, BART(Atr) achieved a MAPE of 16.8 %, an $R^2$ of 0.809 and 70.8 % of predictions within ±20 %, widely outperforming the results reported by Lozano-Gracia and Anselin (2012) for Bogotá and approaching XGBoost, which was the best model. However, the MCMC chains did not converge ($\hat{R} \approx 1.43$), which invalidates the credible intervals for both the predictions and the SHAP values. Finally, the SHAP analysis shows that built area and socioeconomic stratum are the variables most strongly associated with price, followed by geographic features such as distance to *landmarks*.

**Keywords:** Bayesian statistics; BART; adaptive trimming; mass appraisal; asking prices; explainable artificial intelligence (XAI); SHAP values; MCMC; Bogotá.

## Resumen

En Colombia no existe la facilidad para acceder a datos oficiales sobre precios reales de venta de vivienda, y cuando existe, este acceso suele ser poco confiable debido a la subdeclaración fiscal. Bajo este marco, se plantea un prototipo de Sistema de Valuación Masiva (MAS) para la vivienda usada en Bogotá, Soacha, Chía y Zipaquirá, utilizando precios pedidos recolectados mediante *web scraping* del portal FincaRaíz. Como modelo principal se implementa BART(Atr), una adaptación de BART con recorte adaptativo de atípicos propuesta por Cao et al. (2024), el cual se compara mediante validación cruzada contra OLS, Random Forest, XGBoost, LightGBM y un predictor de la mediana. En el conjunto de prueba BART(Atr) obtuvo un MAPE de 16,8 %, un $R^2$ de 0,809 y un 70,8 % de predicciones dentro del ±20 %, superando con creces lo reportado por Lozano-Gracia and Anselin (2012) para Bogotá y acercándose a XGBoost, que fue el mejor modelo. No obstante, las cadenas MCMC no convergieron ($\hat{R} \approx 1{,}43$), lo cual invalida los intervalos de credibilidad de las predicciones y de los valores SHAP. Por último, el análisis SHAP muestra que el área construida y el estrato son las variables de mayor asociación con el precio, seguidas por características geográficas como la distancia a *landmarks*.

**Palabras clave:** Estadística bayesiana; BART; recorte adaptativo de atípicos; valuación masiva de inmuebles; precios pedidos; inteligencia artificial explicable (XAI); valores SHAP; MCMC; Bogotá.

## Orden de lectura recomendado

Se recomienda el siguiente orden de lectura de los cuadernos (todos están alojados en la carpeta `notebooks/`):
1. **Basic_EDA.ipynb:** Contiene un análisis y una contextualización general de los datos, las hipótesis y los objetivos del proyecto.
2. **data_cleaning.ipynb:** Muestra el proceso de análisis de calidad de los datos, así como su limpieza; presenta con su justificación los pasos que posteriormente harán parte del pipeline.
3. **data_preparation.ipynb:** Muestra el preprocesamiento de los datos; justifica cada paso de ingeniería, selección o codificación del pipeline y, además, muestra las relaciones de las características con la variable objetivo.
4. **bivariate_eda.ipynb:** Muestra asociaciones clave entre las características y la variable objetivo; da un preámbulo a los hallazgos hechos con el modelo.
5. **BART_Trees.ipynb:** Implementa el modelo BART(Atr) junto a su pipeline y muestra las métricas y los resultados principales del proyecto.

> Si se desea ahorrar tiempo, el producto final es `BART_Trees.ipynb`; no obstante, también se recomienda leer `Basic_EDA.ipynb` para comprender la generalidad de los datos.

> [!WARNING]
> **No se recomienda ejecutar `BART_Trees.ipynb`:**
> Una ejecución de este cuaderno desde cero puede tomar más de 8 horas; adicionalmente, su caché ocupa aproximadamente 120 GB.

## Estructura

```
BART-trees-for-estimating-housing-prices-in-Bogot-/
├── README.md                       # Este archivo
├── CLAUDE.md                       # Orientación para Claude Code
├── REVIEW_TOS.md                   # Debe leerse antes de hacer scraping a gran escala
├── LICENSE                         # MIT
├── Makefile                        # Control de calidad y comandos del scraper
├── config.py                       # Objeto único de configuración
├── ruff.toml                       # Reglas de PEP 8 (longitud de línea 79)
├── requirements.txt                # Dependencias de ejecución
├── requirements/                   # Dependencias de desarrollo
├── .github/                        # Integración continua (make quality)
├── docs/                           # Artículo final en PDF
├── scraper/                        # Pipeline de extracción de datos
├── tests/                          # 97 pruebas sin conexión del scraper
├── tools/                          # Utilidades que invoca el Makefile
├── data/                           # Conjuntos de datos (excluidos de git)
├── geodata/                        # Landmarks de Bogotá
├── notebooks/                      # Cuadernos de Jupyter
│   ├── Basic_EDA.ipynb
│   ├── data_cleaning.ipynb
│   ├── data_preparation.ipynb
│   ├── bivariate_eda.ipynb
│   ├── BART_Trees.ipynb
│   └── bayesiana_primera_pt.ipynb
├── utils/                          # Funciones auxiliares
│   └── bart_article.py             # Implementación de BARTAtrRegressor
├── src/                            # Reservada para código de modelos
├── outputs/                        # Tablas y figuras
├── cv_cache/                       # Caché de la validación cruzada (excluida de git)
└── out-<operation>-<city>/         # Directorios de trabajo del scraper (excluidos de git)
```


## Diccionario de variables

Las 31 columnas del CSV, en orden.

| Variable | Unidad | Descripción |
|---|---|---|
| `listing_id` | id | Identificador único del anuncio en FincaRaíz. |
| `operation` | categoría | Tipo de transacción anunciada: `venta` o `arriendo`. |
| `property_type` | categoría | Tipo de inmueble: `apartamento`, `casa`, `oficina`, `bodega`, `local`, `lote`, entre otros. |
| `price_amount` | COP | Precio pedido publicado en el anuncio. |
| `price_currency` | categoría | Moneda en la que se expresa el precio pedido. |
| `admin_fee` | COP/mes | Cuota mensual de administración del inmueble. |
| `total_monthly_cost` | COP/mes | Canon de arriendo más administración; definido solo para anuncios de arriendo. |
| `price_per_m2` | COP/m² | Precio pedido dividido entre el área construida. |
| `area_built_m2` | m² | Área construida del inmueble. |
| `area_private_m2` | m² | Área privada interior de la unidad. |
| `bedrooms` | conteo | Número de habitaciones. |
| `bathrooms` | conteo | Número de baños. |
| `parking_spaces` | conteo | Número de parqueaderos incluidos con el inmueble. |
| `stratum` | ordinal 1–6 | Estrato socioeconómico asignado a la dirección. |
| `floor` | número de piso | Piso del edificio en el que se ubica la unidad. |
| `floors_total` | conteo | Número total de pisos del edificio. |
| `age_bracket_code` | ordinal 0–5 | Antigüedad del inmueble por rangos: 0 = a estrenar, 1 = menos de 1 año, 2 = 1–8 años, 3 = 9–15 años, 4 = 16–30 años, 5 = más de 30 años. |
| `age_bracket_label` | categoría | Etiqueta en texto (español) del rango de antigüedad. |
| `construction_state` | categoría | Estado declarado en el anuncio: `Usado`, `Excelente estado`, `En construcción`, entre otros. |
| `city` | nombre | Municipio en el que se ubica el inmueble. |
| `locality` | categoría | Localidad de Bogotá que contiene el inmueble. |
| `neighborhood` | nombre | Barrio del inmueble. |
| `address_text` | texto | Dirección tal como aparece publicada en el anuncio. |
| `latitude` | grados (WGS84) | Latitud de la ubicación del inmueble en el mapa. |
| `longitude` | grados (WGS84) | Longitud de la ubicación del inmueble en el mapa. |
| `amenities` | lista (sep. por `\|`) | Amenidades y características atribuidas al inmueble o a su edificio. |
| `agency_name` | nombre | Inmobiliaria que publica el anuncio. |
| `publisher_type` | categoría | Tipo de anunciante: `inmobiliaria` o `desarrollador`. |
| `published_at` | fecha (AAAA-MM-DD) | Fecha de la primera publicación del anuncio. |
| `updated_at` | fecha (AAAA-MM-DD) | Fecha de la última actualización del anuncio. |
| `source_url` | URL | Dirección web del anuncio. |

El artefacto JSON conserva algunos campos que el CSV aplana, en particular la bandera `price_currency_is_foreign`; por ello, al trabajar con el CSV, se debe revisar la moneda en `price_currency` antes de agrupar precios. Asimismo, conserva `amenities` como objetos `{id, name, group}` en lugar de nombres unidos por `|`.


### Qué contiene cada cuaderno

| Cuaderno | Contenido |
|---|---|
| `Basic_EDA.ipynb` | Contexto, pregunta, hipótesis y primera descripción de los datos. |
| `data_cleaning.ipynb` | Calidad de los datos, patrón de faltantes, imputación y atípicos. |
| `data_preparation.ipynb` | Creación de variables, codificación y VIF. |
| `bivariate_eda.ipynb` | Asociaciones de las variables con el precio y entre sí. |
| `BART_Trees.ipynb` | Pipeline, validación cruzada, MCMC, SHAP y métricas finales. |
| `bayesiana_primera_pt.ipynb` | Primer avance: modelos Beta-Binomial, Multinomial-Dirichlet y Normal-Gamma Inversa. |

## Objeto `BARTAtrRegressor` explicado

Se generó una clase llamada `BARTAtrRegressor`, alojada en [utils/bart_article.py](utils/bart_article.py), para implementar el método de recorte adaptativo de atípicos de Cao et al. (2024). Esta clase se basa en un envoltorio que reentrena árboles BART de la librería `stochtree` y, al seguir la interfaz de scikit-learn, puede usarse dentro de un `Pipeline` o de una validación cruzada.

La idea central consiste en comparar los residuos ordenados del modelo con los que se esperarían si el ruido fuese normal: mientras ambos coincidan, las observaciones se consideran limpias; cuando los residuos observados crecen por encima de lo esperado, las observaciones restantes se tratan como atípicas y se reentrena el modelo sin ellas. A continuación, se detalla su funcionamiento. Después de cada fórmula se incluye el fragmento de [utils/bart_article.py](utils/bart_article.py) que la implementa, sin anotaciones de tipo ni comentarios.

**Paso 1. Entrenar el modelo.** Se entrena el modelo BART inicial $f(x_i)$ para obtener las predicciones $\hat{y}_i = \hat{f}(x_i)$ (`_fit_bart`).

```python
model = self._fit_bart(
    X_arr, y_arr, self.num_chains if initial_is_final else 1
)
```

**Paso 2. Calcular residuos.** Se obtienen los residuos $\hat{\varepsilon}_i = y_i - \hat{y}_i$ y se ordenan por su valor absoluto de menor a mayor:

```math
|\hat{\varepsilon}|_{(1)} \le |\hat{\varepsilon}|_{(2)} \le \cdots \le |\hat{\varepsilon}|_{(n)}
```

```python
resid = y_arr - self._draws(model, X_arr).mean(axis=1)
abs_sorted = np.sort(np.abs(resid))
```

**Paso 3. Estimar la varianza con el 25 % de residuos más pequeños.** Se calcula la estimación robusta $\hat{\sigma}^2$ utilizando únicamente los primeros $\lfloor n/4 \rfloor$ residuos absolutos ordenados, y se corrige la escala usando la distribución teórica (`robust_sigma2` y `folded_normal_quantiles`):

```math
\hat{\sigma}^2 = \frac{\sum_{i=1}^{\lfloor n/4 \rfloor} |\hat{\varepsilon}|_{(i)}^2}{\sum_{i=1}^{\lfloor n/4 \rfloor} \left[ \Phi^{-1}\left( \frac{1 + \tilde{p}_i}{2} \right) \right]^2}
```

donde $\tilde{p}_i = \frac{i}{n+1}$.

```python
def folded_normal_quantiles(n):
    p_tilde = np.arange(1, n + 1) / (n + 1)
    return norm.ppf((1 + p_tilde) / 2)


def robust_sigma2(abs_sorted, z, frac=0.25):
    k = max(1, int(np.floor(len(abs_sorted) * frac)))
    return float(np.sum(abs_sorted[:k] ** 2) / np.sum(z[:k] ** 2))


z = folded_normal_quantiles(n)
sigma2 = robust_sigma2(abs_sorted, z, self.sigma_frac)
```

**Paso 4. Calcular la varianza muestral acumulada.** Para cada $i = 1, \ldots, n$, se obtiene el promedio de los $i$ residuos cuadrados más pequeños (`estimate_p`):

```math
s_i^2 = \frac{1}{i} \sum_{j=1}^{i} |\hat{\varepsilon}|_{(j)}^2
```

```python
i = np.arange(1, len(abs_sorted) + 1)
np.cumsum(abs_sorted**2) / i
```

En el código, los pasos 4 a 6 conforman una sola expresión: el paso 4 es el numerador de `ratio` y el paso 5 su denominador, razón por la cual aquí se muestran como expresiones sueltas.

**Paso 5. Calcular el valor esperado de $s_i^2$.** Se determina la línea de base teórica utilizando los cuantiles de la distribución normal plegada, escalados por $\hat{\sigma}$ (`estimate_p`):

```math
E(s_i^2) = \frac{1}{i} \sum_{j=1}^{i} \xi_j^2, \qquad \xi_j = F^{-1}(\tilde{p}_j) = \hat{\sigma} \cdot \Phi^{-1}\left( \frac{1 + \tilde{p}_j}{2} \right)
```

```python
sigma2 * np.cumsum(z**2) / i
```

Aquí `z` contiene los cuantiles de la normal plegada estándar y $\hat{\sigma}^2$ sale como factor común de la suma, de modo que la expresión equivale a $\frac{1}{i} \sum_{j=1}^{i} \xi_j^2$.

**Paso 6. Definir el conjunto de observaciones válidas.** Se evalúa la condición de tolerancia con el parámetro de sintonización $Q$ ($Q = 1{,}28$) para definir el conjunto $S$ (`estimate_p`):

```math
S = \left\{ i \in \{1, \ldots, n\} : \frac{s_i^2}{E(s_i^2)} \le Q \right\}
```

```python
ratio = (np.cumsum(abs_sorted**2) / i) / (
    sigma2 * np.cumsum(z**2) / i
)
return float(np.sum(ratio <= Q) / len(abs_sorted)), ratio
```

**Paso 7. Rearmar el conjunto de entrenamiento.** Se asigna la función de ponderación $w_i = \psi(\hat{\varepsilon}_i)$ a cada observación para filtrar los atípicos, y se reentrena el modelo solo con las observaciones de peso 1 (bucle de `fit`):

```math
\psi(\hat{\varepsilon}_i) = \begin{cases} 1, & |\hat{\varepsilon}_i| \le |\hat{\varepsilon}|_{(\lfloor n\hat{p} \rfloor)} \\ 0, & |\hat{\varepsilon}_i| > |\hat{\varepsilon}|_{(\lfloor n\hat{p} \rfloor)} \end{cases}
```

```python
p_hat, _ = estimate_p(abs_sorted, z, sigma2, self.Q)
p_hat = max(p_hat, self.p_min)
k_pos = max(1, int(np.floor(n * p_hat)))
threshold = float(abs_sorted[k_pos - 1])
weights = (np.abs(resid) <= threshold).astype(int)

model = self._fit_bart(
    X_arr[weights == 1],
    y_arr[weights == 1],
    self.num_chains if final else 1,
)
```

**Paso 8. Verificar la convergencia.** Se actualiza la proporción estimada $\hat{p}^{(Iter)} = |S|/n$ y se verifica el criterio de convergencia con respecto a la tolerancia $\tau = 10^{-4}$ (bucle de `fit`):

```math
\frac{\left| \hat{p}^{(Iter)} - \hat{p}^{(Iter-1)} \right|}{\hat{p}^{(Iter)}} < \tau
```

```python
final = (
    abs(p_hat - p_old) / p_hat < self.tol
    or it == self.max_iter
)
if final:
    break
p_old = p_hat
```

En el código, `final` se calcula antes del reajuste del paso 7, ya que define cuántas cadenas utiliza ese ajuste; si el criterio no se cumple, el bucle vuelve al paso 2.

El proceso se repite hasta converger o hasta llegar al máximo de iteraciones ($\text{Iter}_{\text{máx}} = 20$).

Los hiperparámetros propios del recorte son los siguientes:

| Parámetro | Valor | Justificación |
|---|---|---|
| `Q` | 1,28 | Umbral de $s_i^2 / E(s_i^2)$ para definir $S$; elegido por Cao et al. (2024) por su balance entre precisión y costo. |
| `trim` | `True` | Activa el recorte duro: los pesos $\psi(\hat{\varepsilon}_i)$ valen 0 o 1 y excluyen los atípicos. |
| `sigma_frac` | 0,25 | Estima $\sigma^2$ solo con el 25 % de residuos más pequeños, Cao et al. (2024). |
| `p_min` | 0,5 | Valor inicial $\hat{p}^{(0)} = 0{,}5$ y cota inferior: se recorta como máximo la mitad de los datos. |
| `tol` | $10^{-4}$ | Tolerancia relativa $\tau$ sobre $\hat{p}$ para detener la iteración, Cao et al. (2024). |
| `max_iter` | 20 | Máximo de iteraciones del recorte, Cao et al. (2024). |

Cabe resaltar dos detalles de la implementación. Por un lado, `p_min` es una salvaguarda que no hace parte del método original: la proporción usada en el recorte es $\hat{p} = \max(|S|/n, p_{min})$, lo cual evita que una iteración inestable elimine casi todos los datos. Por otro lado, únicamente el ajuste final corre con `num_chains` cadenas, ya que los reajustes intermedios solo aportan residuos; de esta manera, el diagnóstico de convergencia se realiza sobre el modelo definitivo con un menor costo computacional.

Su uso es el de cualquier regresor de scikit-learn:

```python
from utils.bart_article import BARTAtrRegressor

modelo = BARTAtrRegressor(
    Q=1.28,
    num_burnin=3000,
    num_mcmc=6000,
    num_chains=2,
    num_gfr=2,
    keep_every=5,
    random_state=42,
).fit(X, y)

modelo.predict(X)  # Media posterior de f(x)
modelo.posterior_draws(X)  # Muestras de f(x): (cadena, muestra, fila)
modelo.p_hat_  # Proporción estimada de datos limpios
modelo.outliers_  # Índices de las filas recortadas
```

## Conclusión y discusión de resultados

En general, los objetivos se cumplieron satisfactoriamente: se logró generar un sistema AVM con alta capacidad explicativa mediante XAI; además, se logró superar con creces el rendimiento de la propuesta de Lozano-Gracia and Anselin (2012) para Bogotá, incluso con el modelo BART (Árboles). Adicionalmente, aunque las métricas de Iban (2022) sean superiores, se considera una ganancia que el modelo BART propuesto se haya acercado significativamente, teniendo en cuenta que el presente proyecto considera toda la ciudad y no solo un sector de esta.

### Conclusiones y trabajo futuro

Se logró interpretar el modelo mediante métodos SHAP y se confirmaron las hipótesis 1, 2, 3 y 5. La variable que indicaba la cercanía al transporte público no fue considerada en la selección final, por lo que la hipótesis 4 no se logró demostrar.

Dado que la cadena MCMC no convergió (a pesar del extenso tiempo de entrenamiento), no se logró cumplir el objetivo específico relacionado con la inferencia bayesiana completa; a pesar de esto, se logró proponer una primera iteración de esta clase de modelos que se acerca al rendimiento predictivo de arquitecturas ya consolidadas en la literatura dentro de un contexto urbano real. Por lo tanto, los trabajos futuros se centrarán en experimentar con distintas distribuciones a priori o en modificar el conjunto de entradas mediante ingeniería de características avanzada, esto con el fin de alcanzar la convergencia de la cadena MCMC y aprovechar con total rigor las propiedades bayesianas del modelo. Además, cabe recordar que una limitación clave del proyecto (y en general en mercados como Bogotá) es la falta de datos sobre los precios de venta, lo cual obliga a utilizar los precios pedidos, tal como lo hace este artículo y otros autores.

## Calidad del código

```bash
make quality     # ruff check y verificación de formato en notebooks, scraper, src y utils
make test        # 97 pruebas sin conexión
make validate    # ambos
make format      # corrección automática
```

`make quality` revisa los cuadernos celda por celda. La longitud de línea es la de PEP 8, 79 caracteres; si dificulta el código de análisis, se puede aumentar en [ruff.toml](ruff.toml).

## Licencia

[MIT](LICENSE) © 2026 Juan Pablo Carrillo Hortua
