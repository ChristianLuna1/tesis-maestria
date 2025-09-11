# Práctica 7 — Ensayo comparativo de optimizadores (DE, GD aproximado y GD analítico)

**Autor:** Christian Luna — Maestría en Cómputo Aplicado  
**Tema:** Comparación de *Differential Evolution (DE)*, *Gradient Descent (GD) aproximado* y *GD analítico* aplicados a **regresión logística** con pérdida **Negative Log-Likelihood (NLL)**.  
**Dataset:https://raw.githubusercontent.com/ChristianLuna1/tesis-maestria/refs/heads/main/Machine_Learning/breast_cancer_wisconsin_diagnostic.csv (1 variable explicativa, etiqueta binaria); partición usada en el cuaderno: `train = 160`, `test = 40` (balanceado 100/100).  
**Semilla:** 42.  

---

## 1. Planteamiento

El objetivo fue estimar los parámetros de un clasificador logístico \(\theta = (w, b)\) minimizando la **NLL** promedio:
\[
\mathcal{L}(\theta)=\frac{1}{n}\sum_{i=1}^n\Big(\operatorname{softplus}(z_i)-y_i z_i\Big),\quad z_i=w^\top x_i+b,
\]
con \(\sigma(z)=1/(1+e^{-z})\).

Trabajé con **tres optimizadores** sobre el mismo modelo y datos:

1. **DE** (*Differential Evolution*): no usa gradientes; busca global con población.  
2. **GD aproximado**: gradiente **numérico** por **diferencia centrada** (no API de ML).  
3. **GD analítico**: gradiente **cerrado** de la NLL para logística.

> Todo el preprocesamiento (z‑score) se calculó en *train* y se aplicó a *test*. 

---

## 2. Configuración de entrenamiento

- **DE**: población `30`, estrategia `rand/1/bin`, `F=0.8`, `CR=0.9`, `800` generaciones, límites `(-3,3)` para cada parámetro.
- **GD aproximado**: `lr0=0.15`, `max_iter=600`, `delta=1e-6`, *learning rate* con decaimiento suave.
- **GD analítico**: `lr0=0.2`, `max_iter=2000`, gradiente cerrado, *learning rate* con decaimiento.

> Todos los tiempos son de la misma sesión. Las cifras pueden variar ligeramente en otras corridas, pero las conclusiones se mantienen.

---

## 3. Resultados cuantitativos (test)

| Método         | NLL_test | LogLoss | Tiempo (s) |
|----------------|---------:|--------:|-----------:|
| DE             | 1.086043 | 0.332270| 29.073618  |
| GD_aprox       | 0.020535 | 0.020535| 4.122071   |
| GD_analítico   | 0.006510 | 0.006510| 0.462615   |

**Observaciones (lectura directa de la tabla):**
- **GD analítico** obtuvo la **mejor NLL** en test (**0.00651**), con el **menor tiempo** de entrenamiento (~0.46 s).
- **GD aproximado** quedó **cerca** (**0.02053**), pero su costo por iteración es mayor porque calcula el gradiente numéricamente (~4.12 s).
- **DE** se quedó **muy por encima** en NLL (**1.0860**) y además requirió **mucho más tiempo** (~29.07 s) debido al enfoque poblacional.



Observaciones:
- Los **tres** optimizadores aprenden un clasificador útil; sin embargo, **GD analítico** alcanza el **menor NLL** en test.
- **GD aproximado** queda **cerca** del analítico, pero con algo más de coste por iteración (cada paso requiere evaluar la pérdida varias veces para estimar el gradiente).
- **DE** reduce la pérdida con rapidez al principio, pero se **estabiliza** antes y queda por arriba en NLL, con un **tiempo total mayor** debido al enfoque poblacional.

---

## 4. Convergencia: ¿qué muestran las curvas?

Incluí dos gráficos en el cuaderno:

### a) NLL vs. iteraciones/generaciones
- **GD analítico** desciende **rápido** y continúa puliendo hasta ~`7.9e-2`. La curva es suave y estable.
- **GD aproximado** sigue una **trayectoria similar** durante las primeras centenas de iteraciones y llega a ~`1.3e-1`. La pequeña diferencia se explica por el **ruido numérico** del gradiente y un presupuesto menor de iteraciones.
- **DE** cae fuerte en pocas generaciones, pero **se “aplana”** alrededor de ~`1.8e-1` con el presupuesto fijado (800 gens).

### b) NLL vs. evaluaciones de la pérdida (costo real)
- **GD analítico** es el **más eficiente** por evaluación: con una única evaluación de NLL por iteración avanza de forma consistente.
- **GD aproximado** necesita **≈2 evaluaciones por parámetro** y por iteración (diferencia centrada), por eso aparece **a la derecha** del analítico en coste, aunque la NLL baje bien.
- **DE** consume **muchas** evaluaciones (población × generaciones). Con el mismo presupuesto, su NLL final queda por encima.

**Conclusión de convergencia (según mis curvas):**
- Por iteraciones: el **GD analítico** desciende rápido y sigue puliendo; el **GD aproximado** replica la forma y converge un poco más arriba; **DE** cae al inicio y se estabiliza pronto.
- Por costo (evaluaciones de NLL): el **GD analítico** logra el **mejor NLL con mucho menos costo**; el **GD aproximado** necesita muchas más evaluaciones; **DE** consume todavía más y termina con NLL mayor.

---

## 5. Ventajas y desventajas (desde este experimento)

### Differential Evolution (DE)
**Ventajas**
- No requiere gradientes → útil en **funciones no derivables**, con **ruido** o **multimodales**.
- Buena **exploración global**: tiende a escapar de óptimos locales.

**Desventajas**
- **Coste alto** en evaluaciones; tiempo mayor.
- En problemas **suaves/convexos** (como este), suele **estancarse** antes que GD.

### GD aproximado (gradiente numérico)
**Ventajas**
- **Plug‑and‑play**: basta con evaluar la pérdida; no hace falta derivar.
- Fácil de implementar y depurar; sirve cuando el gradiente analítico es engorroso.

**Desventajas**
- **Caro por iteración**: requiere varias evaluaciones de la NLL por paso.
- **Sensibilidad** al tamaño de paso \(\delta\); introduce **ruido numérico**.

### GD analítico (gradiente cerrado)
**Ventajas**
- **Eficiencia** por iteración y **convergencia** más **rápida/estable**.
- Escalable a mayor dimensión y datos; aprovecha cálculo vectorizado.

**Desventajas**
- Exige **derivar** la función objetivo (a veces no trivial).  
- Requiere cuidado con el **escalado** y el *tuning* del *learning rate*.

---

## 6. ¿Por qué preferir modelos/optimizadores analíticos cuando es posible?

Cuando el problema admite una formulación **suave** y el gradiente es **accesible**, como la logística con NLL, el **GD analítico** suele:
- **Converger a mejores mínimos** con **menos coste** (una evaluación por iteración).
- Mostrar **trayectorias estables** (menos ruido) y sensibilidad controlable mediante el *learning rate*.
- Ser **más reproducible** y **escalable** en producción.

Los métodos **aproximados** (GD numérico, DE) son valiosos cuando **no hay gradiente**, la función es **no diferenciable** o el paisaje es **muy irregular**. En esos casos, DE puede abrir camino y GD numérico permite optimizar sin derivaciones manuales. Pero **si hay gradiente**, la **ventaja práctica** del analítico en **tiempo/calidad** es clara, como se evidenció en mis resultados.




