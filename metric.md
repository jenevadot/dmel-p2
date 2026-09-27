# Guía Técnica: Elección de Métricas en Entrenamiento y Early Stopping

Este documento analiza de manera formal y empírica cómo la selección de funciones de costo y métricas de evaluación condiciona la convergencia, la dinámica del gradiente, el estadístico predictivo aprendido y los criterios de parada temprana (*early stopping*).

---

## 1. Taxonomía: Función de Pérdida vs. Métrica de Evaluación

En el ciclo de vida del aprendizaje supervisado, las métricas cumplen dos roles complementarios pero con restricciones matemáticas distintas:

* **Función de Pérdida / Costo ($\mathcal{L}$):**
  * Utilizada por el optimizador mediante retropropagación (*backpropagation*).
  * **Requisitos:** Debe ser casi en todo punto diferenciable respecto a los parámetros $\theta$, continua y computationally eficiente sobre mini-lotes (*mini-batches*).
  * **Propósito:** Definir la superficie de error sobre la cual se calcula el gradiente $\nabla_\theta \mathcal{L}$.

* **Métrica de Monitoreo / Validación / Early Stopping:**
  * Empleada para evaluar el desempeño real, calibrar hiperparámetros y detener el entrenamiento antes de sobreajustar (*overfitting*).
  * **Requisitos:** Debe reflejar el objetivo operativo o físico del dominio. No requiere ser diferenciable ni localmente convexa.

---

## 2. Definición Formal de las Métricas

Sea un lote o serie de $n$ observaciones reales $y_i$ y sus correspondientes predicciones $\hat{y}_i$, con residuo $e_i = y_i - \hat{y}_i$:

### 2.1 MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| = \frac{1}{n} \sum_{i=1}^n |e_i|$$

* **Comportamiento analítico:** Crecimiento estrictamente lineal respecto al residuo.
* **Derivada respecto a $\hat{y}$:**
  $$\frac{\partial |e_i|}{\partial \hat{y}_i} = -\text{sign}(y_i - \hat{y}_i) = -\text{sign}(e_i), \quad \forall e_i \ne 0$$
* **Propiedades:** No diferenciable en $e_i = 0$. Invariante en magnitud de gradiente ante grandes desviaciones.

---

### 2.2 Huber Loss (Pérdida de Huber / Smooth $L_1$)

Parametrizada por un umbral $\delta > 0$, interpola suavemente entre una respuesta cuadrática (cerca de cero) y lineal (lejos de cero):

$$L_\delta(e_i) = \begin{cases} 
\frac{1}{2} e_i^2 & \text{si } |e_i| \le \delta \quad (\text{zona cuadrática}) \\ 
\delta \left( |e_i| - \frac{1}{2}\delta \right) & \text{si } |e_i| > \delta \quad (\text{zona lineal}) 
\end{cases}$$

$$\text{Huber Mean} = \frac{1}{n} \sum_{i=1}^n L_\delta(e_i)$$

* **Derivada continua respecto a $\hat{y}$:**
  $$\frac{\partial L_\delta(e_i)}{\partial \hat{y}_i} = \begin{cases} 
  -e_i & \text{si } |e_i| \le \delta \\ 
  -\delta \cdot \text{sign}(e_i) & \text{si } |e_i| > \delta 
  \end{cases}$$
* **Propiedades:** Diferenciable de clase $C^1$. Suavidad cuadrática en el origen y gradiente acotado en $[-\delta, \delta]$.

---

### 2.3 NSE (Nash-Sutcliffe Efficiency)

Métrica estándar en hidrología y ciencias ambientales para evaluar la bondad de ajuste frente a la varianza de la serie observada:

$$\text{NSE} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\text{MSE}}{\text{Var}(y)}$$

donde $\bar{y} = \frac{1}{n}\sum_{i=1}^n y_i$ es la media observada.

* **Rango:** $(-\infty, 1]$.
  * $\text{NSE} = 1.0$: Ajuste exacto ($\text{SS}_{\text{res}} = 0$).
  * $\text{NSE} = 0.0$: El modelo tiene la misma precisión cuadrática que una predicción constante de la media observada $\bar{y}$.
  * $\text{NSE} < 0.0$: El modelo introduce más error cuadrático que el promedio histórico.
* **Propiedades:** Al elevar los residuos al cuadrado en el numerador, penaliza exponencialmente los errores cometidos en los picos y valores extremos.

---

## 3. Impacto en la Dinámica de Entrenamiento

### 3.1 Estimador Estadístico Condicional Aprendido

Cuando una red neuronal minimiza el riesgo empírico bajo una pérdida dada, el valor asintótico que aproxima $\hat{y}(x)$ depende directamente de la formulación de la pérdida:

| Función de Pérdida | Problema Teórico de Optimización | Estimador Convergente $\hat{y}(x)$ |
| :--- | :--- | :--- |
| **MSE / (1 - NSE)** | $\arg\min_{\hat{y}} \mathbb{E}[(Y - \hat{y})^2 \mid X=x]$ | **Media Condicional:** $\mathbb{E}[Y \mid X=x]$ |
| **MAE ($L_1$)** | $\arg\min_{\hat{y}} \mathbb{E}[\|Y - \hat{y}\| \mid X=x]$ | **Mediana Condicional:** $\text{Med}(Y \mid X=x)$ |
| **Huber Loss** | $\arg\min_{\hat{y}} \mathbb{E}[L_\delta(Y - \hat{y}) \mid X=x]$ | **Estimador-M:** Media recortada robusta |

#### Implicancia en distribuciones asimétricas (*skewed*):
* En series hidrológicas, de precipitación o tráfico de red, los eventos extremos son escasos pero de alta magnitud.
* **Entrenar con MAE:** El modelo tiende a pegarse al régimen base y predice valores cercanos a la mediana, ignorando la cola derecha.
* **Entrenar con MSE / NSE:** El modelo desplaza sus estimaciones hacia arriba para minimizar la penalización cuadrática que provocarían los picos subestimados.

---

### 3.2 Comportamiento de los Gradientes durante la Retropropagación

Por la regla de la cadena, para cualquier parámetro $\theta$:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \theta}$$

El término $\frac{\partial \mathcal{L}}{\partial \hat{y}}$ regula la magnitud del paso y la sensibilidad al error:

```
               DERIVADA DE LA PÉRDIDA RESPECTO AL RESIDUO (e = y - ŷ)

         MSE / NSE                    MAE                        Huber (δ)
        (Cuadrático)               (Lineal)                       (Híbrido)

           ∂L/∂e                     ∂L/∂e                         ∂L/∂e
             ^                         ^                             ^
             |    /                    |                             |    ----- +δ
             |   /                     |  +1                         |   /
             |  /                      |                             |  /
      -------+-------> e        -------+-------> e            -------+-------> e
            /|                         |                            /|
           / |                         |  -1                       / |
          /  |                         |                    -δ ---   |
             |                         |                             |
      Gradiente ilimitado      Gradiente fijo (±1)            Clipping intrínseco
      Riesgo de explosión      Oscilaciones en e=0            Estabilidad garantizada
```

* **MAE:** Al tener magnitud constante $|\frac{\partial \mathcal{L}}{\partial \hat{y}}| = 1$, los gradientes no decaen conforme el modelo se aproxima a la solución. Esto produce rebotes alrededor del óptimo, exigiendo un cronograma de tasa de aprendizaje (*learning rate scheduler*) más estricto.
* **MSE / Numerador de NSE:** El gradiente decae suavemente a cero cuando el error es pequeño ($\partial \mathcal{L}/\partial \hat{y} = -2e$), pero puede crecer desmesuradamente en presencia de *outliers*, provocando inestabilidad numérica o reajustes destructivos de pesos.
* **Huber Loss:** Provee un mecanismo automático de *gradient clipping*. Para errores dentro de $[-\delta, \delta]$, los gradientes se reducen suavemente a cero; para errores extremos, se saturan en $\pm\delta$, evitando la explosión de gradientes sin perder diferenciabilidad.

---

### 3.3 ¿Se puede optimizar directamente el NSE?

Optimizar la función de pérdida $\mathcal{L}_{\text{NSE}} = 1 - \text{NSE}$ presenta particularidades:

$$\mathcal{L}_{\text{NSE}} = \frac{\sum_{i=1}^B (y_i - \hat{y}_i)^2}{\sum_{i=1}^B (y_i - \bar{y}_B)^2}$$

1. **Varianza por lote variante:** Si se calcula sobre mini-lotes pequeños ($B$), la varianza muestral $\text{SS}_{\text{tot}}$ fluctúa entre lotes, alterando artificialmente la tasa de aprendizaje efectiva del optimizador.
2. **Equivalencia asintótica:** Si el denominador se calcula sobre todo el conjunto de entrenamiento (o una ventana representativa fija), $\text{SS}_{\text{tot}}$ actúa como una constante escalar $K = 1/\text{Var}(y)$, convirtiendo la pérdida en una versión reescalada de MSE:
   $$\nabla_\theta \mathcal{L}_{\text{NSE}} = \frac{1}{\text{Var}(y)} \nabla_\theta \text{MSE}$$

---

## 4. Impacto en Early Stopping y Selección del Modelo

El criterio de parada temprana monitoriza una métrica en el conjunto de validación para interrumpir el entrenamiento cuando el rendimiento deja de mejorar por $P$ épocas (*patience*).

### 4.1 Conflicto de Objetivos: Régimen Base vs. Picos

Considérese el siguiente caso empírico con 5 observaciones que contienen un pico extraordinario:

* **Observaciones observadas ($y$):** $[10.0, 12.0, 11.0, 13.0, 50.0]$ con media $\bar{y} = 19.2$ y $\text{SS}_{\text{tot}} = 1190.80$.

| Modelo Candidato | Descripción del Ajuste | MAE | Huber ($\delta=2.0$) | NSE |
| :--- | :--- | :---: | :---: | :---: |
| **Modelo A** | Excelente en valores base (`[10.5, 11.5, 11.2, 12.8, 30.0]`), subestima el pico por 20 unidades. | **4.28** | 7.66 | **0.6636** |
| **Modelo B** | Predictor base constante en la media observada ($\bar{y} = 19.2$). | 12.32 | 22.64 | **0.0000** |
| **Modelo C** | Ruidoso en valores base (`[14.0, 8.0, 15.0, 9.0, 48.0]`), pero aproxima el pico con precisión ($48 \approx 50$). | **3.60** | **5.20** | **0.9429** |

#### Consecuencias en Early Stopping:
* **Monitoreando por MAE:** Si se presentan épocas intermedias donde el modelo mejora la exactitud del régimen bajo a costa de aplanar los picos, el criterio de early stopping seleccionará el modelo con menor error mediano, truncando el aprendizaje de eventos extremos.
* **Monitoreando por NSE:** El criterio priorizará modelos que ajusten la cresta del pico, dado que un error residual de $20$ en el pico aporta $400$ a $\text{SS}_{\text{res}}$, colapsando el puntaje de NSE.

---

## 5. Matriz de Decisión y Recomendaciones

| Caso de Aplicación | Función de Pérdida Recomendada | Métrica de Early Stopping | Justificación Técnica |
| :--- | :--- | :--- | :--- |
| **Hidrología / Alerta de Crecidas** | **Huber Loss** o **MSE con ponderación de picos** | **NSE** o **KGE** (*Kling-Gupta*) | El objetivo operativo penaliza severamente el error en picos, pero Huber evita gradientes explosivos en el entrenamiento. |
| **Pronóstico Base / Datos Ruidosos** | **Huber Loss** o **MAE** | **MAE** o **MedAE** | Resistencia a fallas puntuales de sensores o anomalías espurias. |
| **Estimación de Límites / Cuantiles de Riesgo** | **Pinball Loss / Quantile Loss** | **Pérdida Cuantílica Ponderada** | El objetivo es modelar un percentil específico ($p90$, $p95$), no la media ni la mediana. |

### Reglas de Diseño para el Pipeline:
1. **Desacoplar pérdida de métrica:** No es obligatorio entrenar con la misma métrica que se usa para evaluar. Es una práctica estándar entrenar con Huber Loss (estabilidad numérica) y seleccionar el mejor *checkpoint* con NSE (alineación con el dominio físico).
2. **Calibración de $\delta$ en Huber Loss:** Seleccionar $\delta \approx 1.345 \cdot \text{MAD}$ (donde $\text{MAD}$ es la Desviación Absoluta Mediana) para alcanzar un 95% de eficiencia asintótica frente a errores gaussianos con inmunidad a *outliers*.
3. **Evitar NSE por mini-lotes inestables:** Si se desea orientar la pérdida hacia NSE, mantener un denominador fijo calculado sobre la ventana histórica de validación/entrenamiento, no sobre cada mini-lote individual.
