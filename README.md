# Día 9 — Ecosistema Open Source y Modelos Locales

**Lunes 14 de septiembre de 2026** · Práctica Inetum

Entregable del día: **tabla comparativa de modelos open source en Markdown** y
**script de inferencia funcionando**. La parte del enunciado que pedía conectar
con la API REST de la VM de Azure de Inetum y ejecutar el mismo prompt en 2
modelos desplegados **no se ha podido hacer**: no hay acceso a esa VM. Lo que se
entrega en su lugar es un script de inferencia local que sí se ha ejecutado, y su
salida está en `resultados/`. Queda explícitamente marcado en §7 qué se puede y
qué no se puede concluir con esto.

---

## Índice

1. [Cómo elegir un modelo: la decisión de VRAM](#1-cómo-elegir-un-modelo-la-decisión-de-vram)
2. [Tabla comparativa principal](#2-tabla-comparativa-principal)
3. [La familia Gemma 4 en detalle](#3-la-familia-gemma-4-en-detalle)
4. [Comparativa dentro de Gemma 4](#4-comparativa-dentro-de-gemma-4)
5. [Cuantización: GGUF, GPTQ, Q4, Q8](#5-cuantización-gguf-gptq-q4-q8)
6. [Qué midió el script](#6-qué-midió-el-script)
7. [Lo que no se puede concluir](#7-lo-que-no-se-puede-concluir)
8. [Reproducir](#8-reproducir)
9. [Fuentes](#9-fuentes)

---

## 1. Cómo elegir un modelo: la decisión de VRAM

Antes de comparar capacidad hay que saber si el modelo **cabe**. Y ahí está la
primera sorpresa de este día: **la VRAM no la determina el número de parámetros
sino los bytes por peso**.

```
memoria de los pesos (bytes) = parámetros × bytes_por_peso
```

| Precisión | Bytes por peso | Ejemplo: modelo de 8 B |
|---|---:|---:|
| BF16 / FP16 | 2 | 16,0 GB |
| Q8_0 (8 bits) | ~1,06 | 8,5 GB |
| Q5_K_M (5 bits) | ~0,71 | 5,7 GB |
| **Q4_K_M (4 bits)** | **~0,61** | **4,9 GB** |
| Q2_K (2 bits) | ~0,33 | 3,0 GB |

Con eso, la decisión se reduce a una resta:

1. ¿Cuánta VRAM hay? → `nvidia-smi`, o `perfil_hardware.py` de este repositorio.
2. ¿Cuánta deja libre el sistema? **Resta 1–2 GB** para los drivers y el display.
3. Elige el `Q4_K_M` (o `Q5_K_M` si sobra) del modelo más grande que entre.
4. Suma la **KV cache**, que va aparte: es memoria para el contexto y **no está
   en ninguna tabla de pesos**.

### La KV cache, el error clásico

La KV cache guarda las claves y valores de todos los tokens del contexto para no
recalcularlos. Crece de forma **lineal** con los tokens:

| Modelo | 1k tokens | 16k tokens | 128k tokens |
|---|---:|---:|---:|
| Llama 3.1 8B (FP16) | 0,13 GB | 1,95 GB | 15,62 GB |
| Llama 3.1 70B (FP16) | 0,31 GB | 4,88 GB | 39,06 GB |
| Llama 3.1 405B (FP16) | 0,98 GB | 15,38 GB | 123,05 GB |

*(fuente: blog oficial de Llama 3.1, tabla de KV cache)*

Consecuencia práctica que se ve en la tabla comparativa: **Phi-3 Mini declara
128K de contexto, pero a 393 KB por token la ventana completa son 51,5 GB**. No
caben ni en una GPU de 24 GB. Con contexto de 16K son 6,4 GB, que sí.

> El error opuesto también es real: un `Q4_K_M` de 4,9 GB «entra» en una GPU de
> 6 GB, pero al pedir 128K de contexto la KV cache **se come la VRAM que faltaba**.
> Cuantizar el contexto (`--cache-type-k q8_0`) es la palanca complementaria.

---

## 2. Tabla comparativa principal

**10 modelos, 4 familias.** Requisitos del enunciado: nombre, parámetros, VRAM
necesaria, licencia, idiomas y casos de uso.

Parámetros, licencias, descargas y número de parámetros totales se leyeron de la
**API pública de Hugging Face** (`https://huggingface.co/api/models/{id}`, campo
`safetensors.total`) el 2026-09-14, no de las fichas de marketing. Las cifras de
VRAM llevan marca de origen: **[of]** viene de la documentación oficial del
fabricante; **[der]** es derivación propia por la fórmula de la §1, contrastada
con el tamaño real de los ficheros publicados.

### 2.1 Identificación, parámetros y licencia

| Modelo | Org | Params totales | Params efectivos / activos | Contexto | Licencia | ¿Gated? |
|---|---|---:|---:|---:|---|:---:|
| `google/gemma-4-E2B-it` | Google | 5,12 B | **2,3 B** efec. | 128K | Apache 2.0 | No |
| `google/gemma-4-E4B-it` | Google | 8,00 B | **4,5 B** efec. | 128K | Apache 2.0 | No |
| `google/gemma-4-12B-it` | Google | 11,96 B | 11,95 B | 256K | Apache 2.0 | No |
| `google/gemma-4-26B-A4B-it` | Google | 25,81 B | **3,8 B** activos | 256K | Apache 2.0 | No |
| `google/gemma-4-31B-it` | Google | 31,27 B | 30,7 B | 256K | Apache 2.0 | No |
| `meta-llama/Llama-3.1-8B-Instruct` | Meta | 8,03 B | 8,03 B | 128K | **Llama 3.1 Comunitaria** | **Sí, manual** |
| `meta-llama/Llama-3.1-70B-Instruct` | Meta | 70,55 B | 70,55 B | 128K | **Llama 3.1 Comunitaria** | **Sí, manual** |
| `microsoft/Phi-3-mini-128k-instruct` | Microsoft | 3,82 B | 3,82 B | 128K | **MIT** | No |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral AI | 7,25 B | 7,25 B | **32K** | Apache 2.0 | No |
| `mistralai/Mistral-Small-24B-Instruct-2501` | Mistral AI | 23,57 B | 23,57 B | **32K** | Apache 2.0 | No |

> **«Eficetivos» y «activos» no son sinónimos.**
> - **Efectivo** (E2B, E4B): Per-Layer Embeddings. Cada capa tiene su propia tabla
>   de embeddings pequeña. Los embeddings se consultan y ya están: hacen que el
>   cálculo por token sea barato, pero **siguen ocupando RAM**. Por eso E4B
>   necesita 17,9 GB en BF16, no los 9 GB que darían sus 4,5B efectivos.
> - **Activo** (26B-A4B): MoE. Solo se activan 3,8 de los 25,8 B por token. Da
>   velocidad de un 4B, pero **los 25,8 B tienen que estar en memoria** para el
>   enrutamiento. Da memoria de un 26B.

### 2.2 VRAM necesaria

Sólo **pesos**, sin KV cache ni overhead del motor. `≈` marca los valores derivados.

| Modelo | BF16/FP16 | 8 bits | 4 bits (Q4) | Tamaño real del `.gguf` | Hardware mínimo realista |
|---|---:|---:|---:|---:|---|
| `gemma-4-E2B-it` | 11,4 GB [of] | 5,7 GB [of] | **2,9 GB** [of] | 3,12 GB + 0,92 mmproj | Móvil / CPU con 4 GB |
| `gemma-4-E4B-it` | 17,9 GB [of] | 8,9 GB [of] | **4,5 GB** [of] | 4,80 GB + 0,92 mmproj | **GPU de 8 GB** o portátil |
| `gemma-4-12B-it` | 26,7 GB [of] | 13,4 GB [of] | **6,7 GB** [of] | 6,50 GB + 0,16 mmproj | **GPU de 8 GB** |
| `gemma-4-26B-A4B-it` | 57,7 GB [of] | 28,8 GB [of] | **14,4 GB** [of] | 13,45 GB + 1,11 mmproj | GPU de 16 GB |
| `gemma-4-31B-it` | 69,9 GB [of] | 34,9 GB [of] | **17,5 GB** [of] | 16,44 GB + 1,12 mmproj | GPU de 20–24 GB |
| `Llama-3.1-8B-Instruct` | 16 GB [of] | 8 GB FP8 [of] | **4 GB** INT4 [of] | 4,90 GB | **GPU de 6 GB** |
| `Llama-3.1-70B-Instruct` | 140 GB [of] | 70 GB FP8 [of] | **35 GB** INT4 [of] | 42,3 GB a Q4_K_M | 2×RTX 4090 / A100 |
| `Phi-3-mini-128k-instruct` | ≈7,6 GB [der] | ≈4,1 GB [der] | **2,4 GB** descarga [der] | 2,39 GB (oficial MS) | **GPU de 6 GB** o 8 GB RAM |
| `Mistral-7B-Instruct-v0.3` | ≈14,5 GB [der] | ≈7,3 GB [der] | **≈4,4 GB** [der] | ≈4,4 GB (comunidad) | GPU de 6–8 GB |
| `Mistral-Small-24B-Instruct` | ≈47,1 GB [der] | ≈23,6 GB [der] | **≈15 GB** [der] | ≈15 GB (comunidad) | GPU de 16–20 GB |

Los tamaños de `.gguf` seobtained con la API de árbol de ficheros del Hub
(`/api/models/{id}/tree/main`) summing el `.gguf` del modelo y el `mmproj`. En
Gemma 4 el `mmproj` es el proyector multimodal: va ** aparte del LLM**, así que
para imagen o audio hay que sumar esos 0,16–1,12 GB.

**Cruzado con la tabla oficial:** Google publica 4,5 GB para E4B en Q4_0 y el
fichero pesa 4,80 GB; publica 17,5 GB para el 31B y el fichero pesa 16,44 GB. Las
dos cifras concuerdan, así que la tabla oficial es fiable y se puede usar para lo
que todavía no está descargado.

### 2.3 Modalidades

| Modelo | Texto | Imagen | Audio | Video | Genera |
|---|:---:|:---:|:---:|:---:|:---:|
| `gemma-4-E2B-it` | ✅ | ✅ | ✅ | ✅ | texto |
| `gemma-4-E4B-it` | ✅ | ✅ | ✅ | ✅ | texto |
| `gemma-4-12B-it` | ✅ | ✅ | ✅ | ✅ | texto |
| `gemma-4-26B-A4B-it` | ✅ | ✅ | ❌ | ❌ | texto |
| `gemma-4-31B-it` | ✅ | ✅ | ❌ | ❌ | texto |
| `Llama-3.1-8B / 70B` | ✅ | ❌ | ❌ | ❌ | texto |
| `Phi-3-mini-128k` | ✅ | ❌ | ❌ | ❌ | texto |
| `Mistral-7B / Small-24B` | ✅ | ❌ | ❌ | ❌ | texto |

**Gemma 4 es la única familia multimodal de la tabla**, y no de forma uniforme:
el audio desaparece a partir del 12B. Hay una versión aparte, `Phi-3-vision`
(4,2 B, licencia MIT), que no se incluye por no tener equivalente en las otras
tres familias y falsear la comparación.

Detalles que solo importan al working con Gemma 4: el audio admite **30 s
máximos**, el vídeo **60 s a 1 frame/s**, y en el prompt **la imagen va antes del
texto y el audio después**.

### 2.4 Idiomas

| Modelo | Idiomas fuera de caja | Preentrenado en | Fuerte en |
|---|---|---:|---|
| `gemma-4-*` (5 modelos) | 35+ | **140+** | Multilingüe equilibrado |
| `Llama-3.1-8B / 70B` | **8** (en, de, fr, it, pt, **hi**, **es**, th) | 15T+ tokens | Los 8 declarados |
| `Phi-3-mini-128k` | 1 declarado (**inglés**) | 4,9T tokens | Inglés. La ficha **advierte** que otros idiomas rinden peor |
| `Mistral-7B / 24B` | Multilingüe | — | Francés, inglés, español, italiano, alemán |

Para un trabajo en español: **Gemma 4** es la única de las cuatro con cobertura
multilingüe amplia verificada en la ficha (el español está entre los 35+).
Llama 3.1 incluye el español entre sus 8, pero solo 8. Phi-3 Mini **no lo declara
como idioma de uso**.

### 2.5 Casos de uso

| Modelo | Para qué | Cuándo elegirlo |
|---|---|---|
| `gemma-4-E2B-it` | ASR, traducción de voz, descripción de imagen, OCR en móvil | Un móvil. 1,1 GB de RAM en build móvil |
| **`gemma-4-E4B-it`** | Texto + imagen + audio en un portátil de 8 GB | **El equilibrio del día.** Multimodal y cabe en un portátil |
| `gemma-4-12B-it` | Coding y razonamiento con 8 GB de VRAM | Cuando la GPU es de 8–12 GB y se quiere código. Arquitectura sin encoders: multimodal y afinable en una pasada |
| `gemma-4-26B-A4B-it` | Razonamiento avanzado y alto throughput | El que recomienda Google como punto de partida. 3,8 B activos: genera casi tan rápido como un 4B con la capacidad de un 26B |
| `gemma-4-31B-it` | Lo más capaz de la familia | GPU de 24 GB en Q4_0. Mejor benchmark en todo |
| `Llama-3.1-8B-Instruct` | Asistente, resumen de documentos largos, tool calling, código | El más usado. 4,9 GB, GGUF en todas partes |
| `Llama-3.1-70B-Instruct` | Producción con 2×GPU o A100 | Nada de consumo. En una GPU de 24 GB: offload a RAM y 1–3 tok/s |
| `Phi-3-mini-128k` | Móvil y edge, matemática y lógica con poca memoria | Mit y sin gating. 12 tok/s en un iPhone 14 |
| `Mistral-7B-Instruct-v0.3` | Alternativa a Llama 3.1 con Apache 2.0 real | Si quieres licencia libre sin esperar a Meta. **No para documentos largos** (32K) |
| `Mistral-Small-24B-Instruct` | Denso de 24 B en GPU de consumo | Cuando quieres capacidad de 24B sin la complejidad de un MoE |

---

## 3. La familia Gemma 4 en detalle

El modelo del enunciado es `google/gemma-4-E4B-it`, **Apache 2.0**, con **4,5 B
de parámetros efectivos** (8,00 B totales, verificado en el Hub) y **tres
modalidades de entrada**: texto, imagen y audio, generando texto.

### Arquitectura

- **Decoder-only híbrido.** Intercala atención local con ventana deslizante
  (512 tokens en E4B) y atención global completa, con la última capa siempre
  global. Da la velocidad y la huella de un modelo ligero sin perder la
  comprensión a largo plazo.
- **KV cache keywords unificados.** En las capas globales, claves y valores se
  **comparten** (`values = keys`), y se aplica p-RoPE (RoPE proporcional) con
  θ = 0,25 en las globales y RoPE normal en las locales. Efecto publicado:
  **−37,5 % de KV cache global**. Además, E4B comparte KV entre capas con
  ratio 18/42.
- **Per-Layer Embeddings (PLE).** Cada una de las 42 capas tiene su propia tabla
  de embeddings de 2.820 M parámetros. De ahí la diferencia entre 4,5 B
  efectivos y 8 B totales.
- **Encoders separados** (a diferencia del 12B): ~150 M de visión + ~305 M de audio.
- **Vocabulario 262 K** tokens. **35+ idiomas** fuera de caja, preentrenado en 140+.
- **Corte de datos: enero de 2025.**

### Las cuatro cosas que distingue a Gemma 4

1. **Modo thinking.** Razonamiento paso a paso integrated. Se activa con el token
   `<|think|>` al principio del prompt de sistema. Salida con
   `<|channel>thought\n…<channel|>` + respuesta final. Importante: en multi-turno
   el historial debe llevar **solo la respuesta final**, nunca los pensamientos
   anteriores, salvo en turnos con llamadas a herramientas.
2. **Rol de sistema nativo.** Frente a Gemma 3, que lo emulaba, ahora es
   soportado de forma nativa. Es lo que permite controlar el modo thinking.
3. **Resolución de imagen variable.** Presupuesto de tokens visuales
   configurable: **70, 140, 280, 560 o 1120**. Bajarlo para clasificar o
   subtitular vídeo; subirlo para OCR y lectura de letra pequeña.
4. **Function calling nativo.** Pensado para workflows agénticos.

Parámetros de muestreo que recomienda Google de forma estándar:
`temperature=1.0`, `top_p=0.95`, `top_k=64`.

### Checkpoints disponibles

| Sufijo | Qué es | Disponible en |
|---|---|---|
| *(sin sufijo)* | Pesos BF16 | Los 5 |
| `-it` | Instruction-tuned | Los 5 |
| `-unquantized` | Pesos en media precisión del pipeline QAT, para compilación propia o decodificación especulativa | Los 5 |
| `-gguf` | Para llama.cpp / Ollama | Los 5 |
| `-w4a16-ct` | Serializado en `compressed-tensors` para servicio en nube de alta concurrencia | E2B, E4B, 12B, 31B |
| `-mobile-transformers` / `-mobile-ct` | Esquema `wNa8o8` para móvil: capas de decodificación de 2 bits, KV optimizado | Solo E2B y E4B |

QAT = quantization-aware training: **cuantizar no es solo recortar bits**, es
entrenar con la cuantización dentro para que el modelo compense el error. Es la
diferencia entre el `Q4_0` que publica Google (entrenado así) y un `Q4_K_M`
convertido después sin más.

### Código mínimo

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
model = AutoModelForMultimodalLM.from_pretrained(
    "google/gemma-4-E4B-it", dtype="auto", device_map="auto"
)

messages = [
    {"role": "system", "content": "Eres un asistente útil."},
    {"role": "user", "content": "Escribe un chiste corto sobre la RAM."},
]
inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt",
    add_generation_prompt=True, enable_thinking=False,
).to(model.device)

n = inputs["input_ids"].shape[-1]
out = model.generate(**inputs, max_new_tokens=1024)
respuesta = processor.decode(out[0][n:], skip_special_tokens=False)
processor.parse_response(respuesta, prefix=inputs["input_ids"])
```

Para audio e imagen la estructura del prompt cambia — la imagen antes del texto,
el audio después — y se usa la clave `{"type": "audio", "audio": "<url>"}` o
`{"type": "image", "url": "<url>"}`.

---

## 4. Comparativa dentro de Gemma 4

Cifras oficiales de Google para las variantes instruction-tuned. **No son
comparables con las de Phi-3 ni con las de Llama**, porque son harnesses de
evaluación distintos. La columna de Gemma 3 27B está para ver el salto.

| Benchmark | E2B | **E4B** | 12B | 26B A4B | 31B | Gemma 3 27B |
|---|---:|---:|---:|---:|---:|---:|
| MMLU Pro | 60,0 % | **69,4 %** | 77,2 % | 82,6 % | 85,2 % | 67,6 % |
| GPQA Diamond | 43,4 % | **58,6 %** | 78,8 % | 82,3 % | 84,3 % | 42,4 % |
| AIME 2026 (sin herramientas) | 37,5 % | **42,5 %** | 77,5 % | 88,3 % | 89,2 % | 20,8 % |
| LiveCodeBench v6 | 44,0 % | **52,0 %** | 72,0 % | 77,1 % | 80,0 % | 29,1 % |
| Codeforces ELO | 633 | **940** | 1 659 | 1 718 | **2 150** | 110 |
| BigBench Extra Hard | 21,9 % | **33,1 %** | 53,0 % | 64,8 % | 74,4 % | 19,3 % |
| MMMLU (multilingüe) | 67,4 % | **76,6 %** | 83,4 % | 86,3 % | 88,4 % | 70,7 % |
| Tau2 (media de 3) | 24,5 % | **42,2 %** | 69,0 % | 68,2 % | 76,9 % | 16,2 % |
| HLE (sin herramientas) | — | — | 5,2 % | 8,7 % | 19,5 % | — |
| **Visión** | | | | | | |
| MMMU Pro | 44,2 % | **52,6 %** | 69,1 % | 73,8 % | 76,9 % | 49,7 % |
| MATH-Vision | 52,4 % | **59,5 %** | 79,7 % | 82,4 % | 85,6 % | 46,0 % |
| OmniDocBench 1.5 ↓ | 0,290 | **0,181** | 0,164 | 0,149 | **0,131** | 0,365 |
| **Audio** | | | | | | |
| CoVoST ↑ | 33,47 | **35,54** | 38,5 * | — | — | — |
| FLEURS ↓ | 0,09 | **0,08** | 0,069 * | — | — | — |
| **Contexto largo** | | | | | | |
| MRCR v2, 8 needles, 128k | 19,1 % | **25,4 %** | 43,4 % | 44,1 % | **66,4 %** | 13,5 % |

`*` excluyendo chino. ↓ = menor es mejor.

### Qué se lee en esta tabla

- **E4B es la compra racional por debajo de 8 GB.** 69,4 % de MMLU Pro con 4,5 GB
  de pesos es difícil de superar. Gemma 3 27B, con casi seis veces más
  parámetros, saca 67,6 %.
- **El salto de E4B a 12B es enorme en código y razonamiento**: LiveCodeBench
  salta de 52 % a 72 %, y Codeforces de ELO 940 a 1 659. Para un perfil de
  developer, si hay 8 GB, el 12B es mejor opción que el E4B.
- **El 26B A4B se comporta como uno de 30B en capacidad y como uno de 4B en
  velocidad.** Solo pierde frente al 31B en lo de verdad difícil (HLE 8,7 %
  contra 19,5 %) y en contexto largo con 8 needles (44,1 % contra 66,4 %).
- **El salto de E4B a 31B en contexto largo es de 2,6×** (25,4 % → 66,4 %). El
  128K de contexto **declarado** no significa lo mismo en todos los tamaños: hay
  que usar el benchmark para saber cuánto se puede aprovechar.
- **El audio solo existe hasta el 12B**, y aun así los números son modestos:
  CoVoST 38,5 es correcto pero no es ASR de nivel Whisper.

### Phi-3 Mini contra sus comparadores

Cuadro de la propia ficha de Microsoft. Microsoft advierte que sus cifras
«pueden diferir de otros números publicados»; la columna Mistral 7B es la v0.1.
Sirve para las cuatro primeras columnas **entre sí**, no contra Gemma 4.

| Benchmark | Phi-3 Mini | Mistral 7B | Gemma 7B | Llama 3 8B Instruct | GPT-3.5 Turbo |
|---|---:|---:|---:|---:|---:|
| MMLU (5-shot) | **70,9** | 61,7 | 63,6 | 66,5 | 71,4 |
| GSM8K CoT (8-shot) | **85,7** | 46,4 | 59,8 | 77,4 | 78,1 |
| HumanEval (0-shot) | 57,3 | 28,0 | 34,1 | 60,4 | 62,2 |
| MBPP (3-shot) | **69,8** | 50,8 | 51,5 | 67,7 | 77,8 |
| HellaSwag (5-shot) | **75,3** | 58,5 | 49,8 | 71,1 | 78,8 |
| TriviaQA (5-shot) | 61,4 | 75,2 | 72,3 | 67,7 | **85,8** |
| **Media de las 18** | **67,6** | 56,4 | 56,0 | 65,5 | 70,4 |

**Phi-3 Mini (3,8 B, MIT) bate a Llama 3 8B (8 B) con menos de la mitad de
parámetros**, y solo pierde en TriviaQA (61,4 contra 67,7). Es el argumento más
sólido de la tabla a favor de que el número de parámetros no predice la calidad:
importa mucho más la composición del dataset de entrenamiento, que en Phi-3 es
sintético y filtrado con mucho cuidado. La contra es que solo declara inglés.

---

## 5. Cuantización: GGUF, GPTQ, Q4, Q8

### El error conceptual que hay que deshacer

> **GGUF, GPTQ y AWQ no son cosas del mismo tipo.**

GGUF es un **formato de fichero** (un `.gguf` autocontenido que lleva dentro la
arquitectura, el vocabulario, el chat template y los hiperparámetros). GPTQ y AWQ
son **métodos de cuantización**, y su resultado se guarda en un directorio de
`safetensors`. No son alternativas en el mismo eje: se pueden combinar.

### Formatos y métodos

| | **GGUF** | **GPTQ** | **AWQ** |
|---|---|---|---|
| **Qué es** | Formato binario autocontenido | Método de cuantización | Método de cuantización |
| **Cómo funciona** | Cuantización por bloques; se *des*cuantiza al calcular | Optimiza los pesos que minimizan el error de **salida** de cada capa (Hessiana de 2º orden) | Escala por canal los pesos que.importan según las **activaciones**, y absorbe la escala en la capa anterior |
| **Presets** | `Q2_K` … `Q8_0`, `IQ*` | 2/3/4/8 bits, `group_size` 32–128 | 4 bits, `group_size` 32–128 |
| **Bits efectivos** | Q4_K_M ~4,89 · Q5_K_M ~5,70 · Q6_K ~6,56 · Q8_0 ~8,51 | 4,0 + escala | 4,0 + escala |
| **Calibración** | No. Opcional con matriz de importancia | **Obligatoria** (~128 muestras) | **Obligatoria** (~128 muestras) |
| **CPU / Apple Silicon** | ✅ Núcleos optimizados | Solo con GPTQModel | Con kernels de CPU |
| **AMD (ROCm/Vulkan)** | ✅ | Parcial | Parcial |
| **Reparto CPU+GPU** | ✅ Por capas | ❌ | ❌ |
| **Motores** | llama.cpp, Ollama, LM Studio | vLLM, SGLang, Transformers | vLLM, SGLang, Transformers |
| **Calidad a 4 bits** | ~92 % (K-quants) | ~90 % | ~95 % |

**Decisión práctica:** en CPU, Apple Silicon o portátil → **GGUF Q4_K_M** y no hay
discusión. En NVIDIA serviendo a muchos → **AWQ** vía vLLM. En NVIDIA de un solo
usuario y máxima latencia → **EXL3** vía TabbyAPI.

### Los presets de GGUF, en orden

| Preset | Bits | Cuándo |
|---|---:|---|
| `Q2_K` | 2,625 | Último recurso. Degradación muy visible. |
| `Q3_K_M` | 3,91 | Memoria muy ajustada. Inestable en matemáticas. |
| `Q4_0` | 4,55 | Legacy, escala por bloque de 32. **Desaconsejado**: `Q4_K_M` es estrictamente mejor al mismo tamaño. |
| **`Q4_K_M`** | **4,89** | **El punto dulce. La recomendación por defecto.** |
| `Q5_K_M` | 5,70 | Si sobra VRAM, más estable en razonamiento. Suele igualar o superar a un AWQ de 4 bits. |
| `Q6_K` | 6,56 | Casi FP16. |
| `Q8_0` | 8,51 | Línea base sin pérdida apreciable. 8–16 GB para 7B–13B. |

### Por qué `Q4_K_M` mide 4,89 bits y no 4,5

Los K-quants usan superbloques de 256 pesos con escalas y mínimos cuantizados. En
`Q4_K` puro: 256 × 4 = 1 024 bits, más 8 bloques × 12 bits de escala y mínimo
(96), más escala y mínimos de superbloque de 16 bits (32). Total 1 152 ÷ 256 =
**4,5 bits**.

Pero `Q4_K_M` **no es `Q4_K`**: es una *mezcla* que promociona los tensores más
sensibles (`attention.wv` y `feed_forward.w2`) a 6 bits. Por eso el fichero
medio sube a 4,89 bpw. El sufijo indica cuántos tensores se promocionan: `_S`
(pocos), `_M` (el equilibrio), `_L` (muchos).

> **Trampa de nomenclatura:** `Q4_K_M` de dos editores distintos **puede ser un
> fichero diferente** — mismo esquema, distinta versión de llama.cpp al convertir,
> calibración distinta. Si comparas cuantizaciones, compara ficheros del mismo
> editor.

### Lo que la cuantización NO te ahorra

GGUF cuantiza **solo los pesos**. **Las activaciones siguen en coma flotante**, y
la KV cache no se cuantiza salvo que lo pidas explícitamente
(`--cache-type-k q8_0`). Por eso un modelo «de 4,9 GB» con contexto de 128K no
es un modelo de 4,9 GB.

---

## 6. Qué midió el script

Sustituye a la parte del enunciado que pedía la VM de Azure. Todo lo de abajo se
ejecutó de verdad en esta máquina el 2026-09-14.

| Fichero | Qué hace |
|---|---|
| `perfil_hardware.py` | Detecta CPU, RAM, GPU y VRAM; calcula qué modelos de la tabla caben y recomienda cuál |
| `inferencia_local.py` | Inventaria los modelos, ejecuta el mismo prompt en N modelos y mide tokens/s, latencia y longitud |
| `modelos_open_source.json` | Los datos de la tabla, con la fuente de cada cifra |
| `resultados/perfil_hardware.json` | Salida del perfilado en esta máquina |
| `resultados/comparativa_inferencia.md` | Los prompts ejecutados, las respuestas y las métricas |

### 6.1 El hardware de esta máquina

```
Windows 11 · Python 3.14.7 · Intel Family 6 Model 140 (8 núcleos)
RAM total 15,7 GB (libre 4,4 GB) · SIN GPU · Ollama 0.35.0
```

**No hay GPU.** El techo no son los 11 GB de RAM sino el **ancho de banda de la
RAM**, y eso se nota en las cifras de abajo. Con 15,7 GB y una reserva de 2 GB,
el techo son 11,3 GB utilizables, y la recomendación a contexto 8 192 sale así:

| Modelo | Preset | Pesos | KV cache | mmproj | Total | Veredicto |
|---|---|---:|---:|---:|---:|---|
| `gemma-4-12B-it` | Q5_K_M | 7,94 | 1,53 | 0,16 | 9,63 | sí |
| `Llama-3.1-8B-Instruct` | Q5_K_M | 5,33 | 1,03 | — | 6,36 | sí |
| `gemma-4-E4B-it` | Q5_K_M | 5,31 | 1,02 | 0,92 | 7,25 | sí |
| `Mistral-7B-Instruct-v0.3` | Q5_K_M | 4,81 | 0,93 | — | 5,74 | sí |
| `gemma-4-E2B-it` | Q5_K_M | 3,40 | 0,66 | 0,92 | 4,98 | sí |
| `Phi-3-mini-128k-instruct` | Q5_K_M | 2,54 | 0,49 | — | 3,03 | sí |
| `Mistral-Small-24B-Instruct` | (Q2_K) | 7,20 | 3,02 | — | 10,22 | solo degradado |
| `gemma-4-26B-A4B-it` | (Q2_K) | 7,88 | 3,30 | 1,11 | 12,30 | solo degradado |
| `Llama-3.1-70B-Instruct` | — | — | 9,03 | — | — | no cabe |
| `gemma-4-31B-it` | — | — | 4,00 | 1,12 | — | no cabe |

`Total = pesos + KV cache + mmproj`, y la suma cuadra fila a fila. Por ejemplo
E4B: `5,31 + 1,02 + 0,92 = 7,25 GB`.

La **columna `mmproj`** es el proyector multimodal de Gemma 4, que va en un
fichero aparte del LLM: 0,92 GB en E2B y E4B, 1,11–1,12 GB en el 26B y el 31B.
La tabla oficial de Google (4,5 GB para E4B en Q4_0) **no lo incluye**, así que
sumar la columna de VRAM de la §2.2 con el `mmproj` de esta tabla da el total
real que hay que tener en la GPU si se va a usar imagen o audio.

Bajar el contexto de 8 192 a 4 096 tokens mueve la recomendación de `Q5_K_M` a
`Q4_K_M` (E4B pasa de 7,25 a 5,98 GB) y hace entrar al 31B en la zona degradada.
**La KV cache decide tanto como los pesos.**

### 6.2 La inferencia que sí se ejecutó

Un único modelo local: `llama3.1:8b` (Ollama, id `46e0c10c039e`, **4,58 GB**,
Q4_K_M). Es el único del conjunto descargado, así que no hay comparación posible
entre modelos.

| Prompt | Tokens | Palabras | Latencia | tok/s | Verificación automática |
|---|---:|---:|---:|---:|---|
| Razonamiento (descuentos) | 135 | 85 | 56,58 s | 2,39 | **3/3 valores** (64, 78, 58 EUR) |
| Código (máx. con empates) | 70 | 24 | 36,96 s | 1,89 | Bloque de código, solo código, 4 líneas |
| Instrucción (3 frases imperativas) | 57 | 33 | 28,26 s | 2,02 | 3 líneas, numeradas, empiezan por verbo |

Lo que produjo (texto íntegro en `resultados/comparativa_inferencia.md`):

- **Razonamiento:** 80 × 0,80 = 64, 120 × 0,65 = 78, ahorro 16 + 42 = 58. Los
  tres valores correctos, con el desglose paso a paso.
- **Código:** función que calcula `max(...)` y filtra todos los empates, no solo
  el primero. El enunciado pedía devolver *todos* los empatados y eso es
  exactamente lo que hace. Cuatro líneas, sin prosa alrededor.
- **Instrucción:** tres frases numeradas, las tres en imperativo
  (`Ven`, `Observa`, `Explora`), sin relleno ni introducción.

**El dato más útil del día no es la calidad, es el coste: ~2 tokens por
segundo.** Un modelo de 8 B en CPU sin GPU. Es la cifra que decide si un modelo
local es viable para un uso real: generar 500 tokens son más de cuatro minutos.
Con GPU, el mismo modelo va entre 20 y 60 tok/s, o sea 25 veces más rápido.

### 6.3 Por qué la sustitución es legítima

El objetivo del punto del enunciado era **ver qué modelos hay desplegados y
ejecutar el mismo prompt en dos de ellos para analizar la calidad**. Eso es lo
que hace `inferencia_local.py`, con dos diferencias que se documentan en vez de
ocultarse:

1. Donde Azure tendría modelos remotos, aquí hay modelos locales. Por tanto se
   mide además **memoria y velocidad**, no solo la calidad.
2. **Solo hay un modelo.** Con uno no hay comparación, y el script lo dice en su
   propia salida en lugar de disimularlo.

El script habla los dos protocolos, así que el punto del enunciado se ejecuta sin
cambiar una línea en cuanto haya URL:

```powershell
# Hoy: local
python inferencia_local.py --comparar

# Cuando haya VM: el mismo script, otro transporte
python inferencia_local.py --endpoint https://<vm>:8000/v1 --api-key <clave> --comparar
```

Ambos caminos están probados: `--endpoint` contra un servidor inaccesible da un
error legible con la URL que probó, y contra un endpoint que devuelve 200 con
una forma inesperada avisa de qué claves recibió en vez de reventar.

### 6.4 Tres bugs encontrados al ejecutar (no al leer)

Se documentan porque son el contenido real de la sesión, y porque son el mismo
tipo de error que el §5.4 del README de los días 5–7: **instrumentos que
fallan en silencio o que miden lo que no es.**

| Bug | Síntoma | Por qué importa |
|---|---|---|
| Clave `tokens_generados` en el diccionario, `tokens_generated` en la lectura | `KeyError` en el primer prompt | Es un desajuste español/inglés de una línea. Pasó porque el `KeyError` decía `'tokens_generados'` y el mensaje de la línea leída decía `tokens_generated`: **casi el mismo texto**, y eso despistó durante varias iteraciones. La moraleja es que hay que mirar el *valor* de la clave que falta, no solo su nombre en pantalla. |
| `Elecciones` en el comentario, `elecciones` en el código | `NameError` al usar `--endpoint` | Mismo origen. Solo se apareció al probar el segundo transporte, que es la razón de existir dos clientes. |
| La lógica de recomendación elegía el **mayor** modelo que cupiera, aunque fuera a Q2_K | Recomendaba `Mistral-Small-24B` a Q2_K, el peor cuantizado de la lista | El bug conceptual más importante de los tres. **«Cabe en disco» y «sirve bien» no son lo mismo**, y el script confundía las dos cosas. Se corrigió añadiendo un suelo de calidad (`Q4_K_M`) y marcando como *degradado* lo que solo entra por debajo, sin recomendarlo. |

El tercero es el que más conviene studiar: es un fallo de criterio, no de
sintaxis, y un script que dice «este equipo puede con el modelo de 24 B» sin
avisar de que va a 2,6 bits por peso está affirmando algo falso.


---

## 7. Lo que no se puede concluir

Esta sección existe porque un informe de esta tabla sin límites sería engañoso.

| Afirmación tentadora | Por qué **no** se puede decir |
|---|---|
| «Gemma 4 E4B es mejor que Llama 3.1 8B» | **No hay benchmark común.** Las cifras de Gemma 4 son de Google; las de Phi-3, de Microsoft. Harnesses distintos, prompts distintos. Comparar la fila de la §4 con la tabla de la §4.1 es un error metodológico, no una aproximación. |
| «E4B (4,5 B) supera a Llama 3.1 8B» | Por lo mismo. El número de parámetros efectivos y el total son cosas distintas, y la memoria se mide con el total. |
| «Phi-3 Mini bate a Llama 3 8B» | Esto sí es defendible **dentro de la tabla de Microsoft**, que usa el mismo pipeline para las cuatro columnas. Fuera de ella, no. Y Microsoft avisa de lo mismo. |
| «Gemma 4 26B-A4B necesita 14 GB» | Es lo que ocupan los pesos. Los 25,8 B completos están en RAM para el enrutamiento: **3,8 B activos no significa memoria de 3,8 B**. |
| «E4B cabe en un portátil» | Los pesos sí (4,5 GB en Q4_0). El `mmproj` de imagen y audio son 0,92 GB **aparte**, y la KV cache de 128K son varios GB más. |
| «Ejecuté el mismo prompt en 2 modelos y compará la calidad» | **No se ha hecho.** Solo hay 1 modelo local. No hay verdad terreno anotada, ni N repeticiones, ni varianza. Un prompt único a un modelo único no es un benchmark: es una anecdotal, exactamente la trampa que ya se documentó en el §4.4 del README de los días 5–7. |
| «Los modelos de la tabla son open source» | **Llama 3.1 no lo es.** Es una licencia propietaria con condiciones (atribución «Built with Llama», nombre de derivados, umbral de 700 M de usuarios). Y está *gated*: hay que pedir aprobación. Las otras tres familias sí son open source en el sentido OSI (Apache 2.0 / MIT). |
| «Gemma 4 es Apache 2.0, sin letra pequeña» | Correcto para los pesos, y es una ventaja real frente a Llama 3.1. Pero hay que leer la ficha igual: hay gated=false peroFiltering de CSAM, datos sensibles y audio limitado a 30 s. |

**Lo que sí se puede concluir:** que la VRAM la fija el formato y la precisión,
no el nombre del modelo; que `Q4_K_M` es el punto dulce por defecto; que la KV
cache puede superar a los propios pesos; que Gemma 4 es la única familia
multimodal del conjunto y que E4B es su punto dulce por debajo de 8 GB; que
Phi-3 Mini demuestra que 3,8 B pueden ir por delante de 8 B; y que el hardware de
esta máquina es el cuello de botella real.

---

## 8. Reproducir

```powershell
# Requisito: Python 3.10+ y Ollama (https://ollama.com)
ollama list                      # ver qué modelos hay locales
ollama pull llama3.1:8b          # el único verificado en este día

python perfil_hardware.py        # ¿qué cabe en esta máquina?
python inferencia_local.py --listar
python inferencia_local.py --modelo llama3.1:8b
python inferencia_local.py --comparar --temperatura 0.7
python inferencia_local.py --comparar --guardar resultados/
```

Opciones útiles:

```
--modelo X[,Y]     Modelos concretos a comparar
--prompt "texto"   Prompt propio en vez de los 3 de ejemplo
--temperatura F    0 = determinista, 1 = por defecto del modelo
--max-tokens N     Límite de tokens generados
--guardar DIR      Escribe un informe Markdown con las respuestas
--json             Salida por consola en JSON, para procesar
```

Sin dependencias externas: los scripts usan solo la biblioteca estándar y la API
REST de Ollama en `http://localhost:11434`. `requirements.txt` está vacío a
propósito — el propósito de este día es ver el ecosistema funcionando sin
capas de abstracción.

---

## 9. Fuentes

**Datos verificados por API el 2026-09-14:**

- `https://huggingface.co/api/models/{id}` — parámetros (`safetensors.total`),
  licencia, gating, descargas, likes
- `https://huggingface.co/api/models/{id}/tree/main` — tamaño real de los
  ficheros `.gguf` y `mmproj`

**Documentación de fabricantes:**

- Gemma 4 model card — <https://huggingface.co/google/gemma-4-E4B-it>
- Gemma 4 overview y tabla de memoria — <https://ai.google.dev/gemma/docs/core>
- Gemma 4 Technical Report (arXiv 2607.02770) — <https://arxiv.org/html/2607.02770v1>
- Llama 3.1: sizes, VRAM y KV cache — <https://huggingface.co/blog/llama31>
- Phi-3 Mini-4K-Instruct — <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct>
- Phi-3 Mini GGUF oficiales — <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf>
- Phi-3 Technical Report — <https://arxiv.org/html/2404.14219v4>
- Mistral 7B Instruct v0.3 — <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>

**Sobre cuantización:**

- Presets y bits por peso (tabla de HF) y guía de llama.cpp
- *Which Quantization Should I Use?* — arXiv 2601.14277
- Tabla de medidas reales `Q4_K_M`/`Q8_0` sobre Llama-3.1-8B: tools/quantize/README.md

**Recursos del enunciado:**

- Hugging Face NLP Course — <https://huggingface.co/learn/nlp-course/>
- Cisco Skills for All, Data Analytics Essentials —
  <https://skillsforall.com/course/data-analytics-essentials>
- Cisco Skills for All, Computer Hardware Basics —
  <https://skillsforall.com/course/computer-hardware-basics>

# Dia_5-10-2026
