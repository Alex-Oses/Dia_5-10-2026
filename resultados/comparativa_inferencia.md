# Comparativa de inferencia — Día 9 (2026-10-05T11:01:31+02:00)

Backend: **Ollama**. Temperatura 0.7, máximo 512 tokens.

## Métricas

| Modelo | Prompt | Tokens gen. | Palabras | Latencia | tok/s |
|---|---|---:|---:|---:|---:|
| `llama3.1:8b` | razonamiento | 135 | 85 | 56.58 s | 2.39 |
| `llama3.1:8b` | codigo | 70 | 24 | 36.96 s | 1.89 |
| `llama3.1:8b` | instruccion | 57 | 33 | 28.26 s | 2.02 |

## Verificación automática

| Modelo | Prompt | Comprobación | Resultado |
|---|---|---|---|
| `llama3.1:8b` | razonamiento | 3/3 valores (64, 78, 58 EUR) | correcto |
| `llama3.1:8b` | codigo | bloque=True · solo código=True · 4 líneas | cumple |
| `llama3.1:8b` | instruccion | 3 líneas · numeradas=True · exactamente 3=True | cumple |

## Respuestas completas

### razonamiento — llama3.1:8b

```text
**Producto A:**
- 80 EUR * 20% = 16 EUR (descuento)
- 80 EUR - 16 EUR = 64 EUR (precio final)

**Producto B:**
- 120 EUR * 35% = 42 EUR (descuento)
- 120 EUR - 42 EUR = 78 EUR (precio final)

**Ahorro total:**
- 16 EUR + 42 EUR = 58 EUR

**Resultado final:**
El precio final del producto A es de 64 EUR y el precio final del producto B es de 78 EUR. El ahorro total es de 58 EUR.
```

### codigo — llama3.1:8b

```text
```python
def nombre_alumno_nota_mas_alta(alumnos):
    max_nota = max(alumno['nota'] for alumno in alumnos)
    alumnos_con_nota_maxima = [alumno['nombre'] for alumno in alumnos if alumno['nota'] == max_nota]
    return alumnos_con_nota_maxima
```
```

### instruccion — llama3.1:8b

```text
1. Ven a disfrutar de las playas de la costa asturiana.
2. Observa cómo las olas del mar Cantábrico rompen en la costa.
3. Explora la riqueza de biodiversidad que alberga este mar.
```

## Qué se puede y qué no se puede concluir

**Sí:** el coste de la respuesta. Tokens generados, latencia y throughput
son medidas directas y repetibles. En esta máquina el modelo corre sobre CPU,
así que la cifra de tok/s es la de un equipo sin GPU dedicada.

**No:** que un modelo sea mejor que otro. Un prompt por modelo, una sola
ejecución, sin verdad terreno anotada y sin repeticiones no es un
benchmark. La verificación automática comprueba la *forma* de la respuesta
(3 números correctos, que sea código y no prosa, que haya 3 frases), no su
corrección semántica.

**La comparación entre modelos del enunciado (Azure) no se ha hecho.**
No hay modelos desplegados a los que acceder. Con más de un modelo local
el script ya produciría la tabla comparativa; con un endpoint compatible
con OpenAI (`--endpoint`), produciría exactamente el entregable original.
