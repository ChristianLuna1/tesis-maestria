Comparación de optimizadores: DE vs. GD aproximado vs. GD analítico
====================================================================
Problema: clasificación binaria con regresión logística (función objetivo: negative log-likelihood, NLL).
Dataset:
Métricas: NLL en test (principal), Accuracy y AUC (complementarias).
Implementación: tres optimizadores sobre el mismo modelo y datos:
  1) Differential Evolution (DE)
  2) Gradient Descent (GD) aproximado con gradiente numérico (diferencia centrada)
  3) Gradient Descent (GD) analítico con gradiente cerrado

1) Modelo y función objetivo
----------------------------
Regresión logística con parámetros θ = (w, b). Para una muestra x:
    pθ(y=1 | x) = σ(z) = 1 / (1 + exp(-z)), con z = w^T x + b.

Pérdida promedio (Negative Log-Likelihood, NLL) sobre n ejemplos:
    L(θ) = (1/n) * Σ_i [ softplus(z_i) - y_i * z_i ], con z_i = w^T x_i + b.

Gradiente cerrado (GD analítico):
    ∇_w L = (1/n) X^T (σ(Xw+b) - y),    ∇_b L = (1/n) Σ_i (σ(z_i) - y_i).

Gradiente numérico (GD aproximado, diferencia centrada):
    ∂L/∂θ_k ≈ [L(θ + δ e_k) - L(θ - δ e_k)] / (2δ).

2) Configuración experimental
-----------------------------
• Preprocesamiento: normalización z-score calculada en train y aplicada a test.
• Entrenamiento:
  - DE: población 5*(d+1), estrategia rand/1/bin, 800 generaciones.
  - GD aproximado: gradiente centrado, learning rate con decaimiento.
  - GD analítico: gradiente cerrado, learning rate con decaimiento.
• Evaluación: NLL, Accuracy y AUC en test. Curvas de convergencia (escala log).

3) Resultados (test)
--------------------
| Método   | NLL_test | LogLoss |  ACC |  AUC | Tiempo (s) |
|----------|---------:|--------:|-----:|-----:|-----------:|
| DE       |  0.1801  |  0.1801 | 1.00 | 1.00 |    4.0160  |
| GD_aprox |  0.1304  |  0.1304 | 1.00 | 1.00 |    0.2404  |
| GD_an    | *0.0795* | *0.0795*| 1.00 | 1.00 |    0.3101  |

(Estos son los valores observados en nuestra corrida;

4) Convergencia observada
-------------------------
• GD analítico: mejor convergencia; desciende rápido y sigue afinando hasta el menor NLL.
• GD aproximado: trayectoria similar al analítico al inicio; alcanza un NLL cercano, con ligera desventaja por el ruido numérico (aun con diferencia centrada).
• DE: reduce la pérdida con pocos pasos iniciales (exploración) y después se estabiliza; su NLL final es mayor con el mismo presupuesto.

Al graficar NLL vs. número de evaluaciones de la NLL (costo real), GD analítico domina: por evaluación logra mayor reducción de pérdida. GD aproximado es competitivo pero requiere ≈2 evaluaciones por componente para estimar el gradiente; DE consume más evaluaciones totales (población × generaciones).

5) Ventajas y desventajas
-------------------------
Differential Evolution (DE)
• Ventajas: no requiere gradiente (aplicable a funciones no derivables/ruidosas/multimodales); buena exploración global.
• Desventajas: costo alto (muchas evaluaciones); convergencia más lenta en problemas suaves como la logística.

GD aproximado (gradiente numérico)
• Ventajas: “plug-and-play”; solo necesita evaluar la pérdida.
• Desventajas: caro por iteración (O(d) evaluaciones por paso con diferencia centrada); sensible a δ; introduce ruido numérico.

GD analítico (gradiente cerrado)
• Ventajas: más eficiente por iteración; convergencia rápida y estable; escalable.
• Desventajas: requiere derivar la función objetivo (no siempre trivial).

6) Conclusiones
---------------
En regresión logística con NLL (suave y bien condicionada), GD analítico es preferible: menor NLL con menor costo y convergencia más nítida. GD aproximado es una alternativa válida cuando no se dispone del gradiente cerrado; su desempeño aquí es cercano, a costa de más evaluaciones. DE es útil cuando el paisaje es no diferenciable o multimodal; en logística estándar, suele quedar por detrás con el mismo presupuesto.

