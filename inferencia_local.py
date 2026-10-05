"""Inferencia local: inventario de modelos y comparacion de un mismo prompt.

Dia 9 - Ecosistema Open Source y Modelos Locales.

Este script sustituye al punto del enunciado que pedia enumerar los modelos
desplegados en la VM de Azure de Inetum y ejecutar el mismo prompt en dos de
ellos. La VM no ha estado accesible, asi que el defecto es local (Ollama).

Pero el script no es una maqueta: habla los dos protocolos. Con --ollama usa la
API REST de Ollama, que es lo que hay en esta maquina. Con --endpoint habla
cualquier endpoint compatible con OpenAI (vLLM, llama.cpp server, TGI y tambien
Azure), que es exactamente el protocolo que exponen los despliegues de Azure.
En cuanto haya URL, el punto del enunciado se ejecuta sin tocar el codigo.

Medir no es solo pedir texto: ademas mide tokens por segundo, latencia y
longitud, porque en modelos locales el coste de la respuesta es tan parte del
resultado como lo que dice.

Uso:
    python inferencia_local.py --listar
    python inferencia_local.py --modelo llama3.1:8b
    python inferencia_local.py --comparar --guardar resultados/
    python inferencia_local.py --endpoint https://mi-vm:8000/v1 --api-key ...

Solo biblioteca estandar. No hay requirements porque el proposito del dia es
ver el ecosistema sin capas de abstraccion.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Los tres prompts son de dominio distinto a proposito: si dos modelos dieran
# lo mismo en los tres, el prompt no estaria midiendo nada.
PROMPTS_EJEMPLO = [
    {
        "id": "razonamiento",
        "dominio": "razonamiento con calculo",
        "texto": (
            "Una tienda vende el producto A con un 20% de descuento y el producto B "
            "con un 35%. Si A cuesta 80 EUR y B cuesta 120 EUR, cual es el precio "
            "final de cada uno y cuanto se ahorra en total? Responde solo con los "
            "calculos y el resultado final."
        ),
    },
    {
        "id": "codigo",
        "dominio": "generacion de codigo",
        "texto": (
            "Escribe una funcion en Python que reciba una lista de diccionarios "
            "con las claves 'nombre' y 'nota', y devuelva el nombre del alumno "
            "con la nota mas alta. Si hay empate, devuelve todos los "
            "empatados. Solo el codigo, sin explicaciones."
        ),
    },
    {
        "id": "instruccion",
        "dominio": "seguimiento de instruccion restrictiva",
        "texto": (
            "Escribe exactamente 3 frases sobre el mar Cantabrico. Cada frase "
            "debe empezar por un verbo en imperativo. Responde solo con las 3 "
            "frases, numeradas, y nada mas."
        ),
    },
]


# --------------------------------------------------------------------------
# Capa HTTP
# --------------------------------------------------------------------------

class ClienteError(Exception):
    """Error de la capa HTTP, con el detalle que devuelve el servidor."""


def _http_json(
    url: str,
    metodo: str = "GET",
    cuerpo: dict | None = None,
    cabeceras: dict | None = None,
    timeout: int = 600,
) -> dict:
    datos = json.dumps(cuerpo).encode("utf-8") if cuerpo is not None else None
    cabeceras = dict(cabeceras or {})
    if datos is not None:
        cabeceras["Content-Type"] = "application/json"

    peticion = urllib.request.Request(url, data=datos, headers=cabeceras, method=metodo)
    try:
        with urllib.request.urlopen(peticion, timeout=timeout) as respuesta:
            crudo = respuesta.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detalle = exc.read().decode("utf-8", errors="replace")[:400]
        raise ClienteError(f"HTTP {exc.code} en {url}: {detalle}") from exc
    except urllib.error.URLError as exc:
        raise ClienteError(f"No se pudo conectar con {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ClienteError(f"Tiempo agotado esperando a {url}") from exc

    try:
        return json.loads(crudo)
    except json.JSONDecodeError as exc:
        raise ClienteError(f"Respuesta no JSON de {url}: {crudo[:200]}") from exc


class ClienteOllama:
    """API REST de Ollama (http://localhost:11434 por defecto)."""

    nombre = "Ollama"

    def __init__(self, host: str = "http://localhost:11434") -> None:
        self.host = host.rstrip("/")

    def disponible(self) -> bool:
        try:
            self.listar_modelos()
            return True
        except ClienteError:
            return False

    def listar_modelos(self) -> list[dict]:
        datos = _http_json(f"{self.host}/api/tags", timeout=15)
        salida = []
        for m in datos.get("models", []):
            detalles = m.get("details") or {}
            salida.append(
                {
                    "id": m.get("name") or m.get("model"),
                    "parametros": detalles.get("parameter_size") or "?",
                    "cuantizacion": detalles.get("quantization_level") or "?",
                    "familia": detalles.get("family") or "?",
                    "tamano_gb": round(m.get("size", 0) / 1024**3, 2),
                }
            )
        return salida

    def generar(self, modelo: str, prompt: str, temperatura: float, max_tokens: int) -> dict:
        cuerpo = {
            "model": modelo,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperatura,
                "num_predict": max_tokens,
            },
        }
        datos = _http_json(f"{self.host}/api/generate", "POST", cuerpo)
        eval_count = datos.get("eval_count") or 0
        eval_ns = datos.get("eval_duration") or 0
        return {
            "texto": datos.get("response", ""),
            "tokens_generados": eval_count,
            # eval_duration viene en nanosegundos.
            "tok_por_s": round(eval_count / (eval_ns / 1e9), 2) if eval_ns else None,
            "tokens_prompt": datos.get("prompt_eval_count") or 0,
        }


class ClienteOpenAICompatible:
    """Endpoint compatible con OpenAI: vLLM, llama.cpp server, TGI, Azure.

    Es el protocolo que exponen los despliegues de modelos en Azure, asi que
    este cliente es el que ejecutaria el punto del enunciado contra la VM.
    """

    nombre = "OpenAI-compatible"

    def __init__(self, base_url: str, api_key: str = "", modelo_por_defecto: str = "") -> None:
        self.base = base_url.rstrip("/")
        if not self.base.endswith("/v1"):
            self.base = f"{self.base}/v1"
        self.api_key = api_key
        self.modelo_por_defecto = modelo_por_defecto
        self._modelos: list[dict] = []

    def _cabeceras(self) -> dict:
        cab = {"Content-Type": "application/json"}
        if self.api_key:
            cab["Authorization"] = f"Bearer {self.api_key}"
        return cab

    def disponible(self) -> bool:
        try:
            self.listar_modelos()
            return True
        except ClienteError:
            return False

    def listar_modelos(self) -> list[dict]:
        if self._modelos:
            return self._modelos
        datos = _http_json(f"{self.base}/models", cabeceras=self._cabeceras(), timeout=30)
        salida = []
        for m in datos.get("data", []):
            salida.append(
                {
                    "id": m.get("id"),
                    "parametros": "?",
                    "cuantizacion": "?",
                    "familia": "?",
                    "tamano_gb": None,
                    "creado": m.get("created"),
                }
            )
        self._modelos = salida
        return salida

    def generar(self, modelo: str, prompt: str, temperatura: float, max_tokens: int) -> dict:
        cuerpo = {
            "model": modelo,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperatura,
            "max_tokens": max_tokens,
            "stream": False,
        }
        datos = _http_json(
            f"{self.base}/chat/completions", "POST", cuerpo, self._cabeceras()
        )
        texto = ""
        # Un endpoint mal configurado puede devolver 200 con una forma distinta.
        # Se comprueba en vez de asumir, para que el fallo sea legible.
        elecciones = datos.get("choices") or []
        if not elecciones:
            raise ClienteError(
                "La respuesta no trae 'choices'. ¿La URL apunta a un endpoint "
                f"compatible con OpenAI? Claves recibidas: {sorted(datos)[:8]}"
            )
        mensaje = elecciones[0].get("message") or {}
        texto = mensaje.get("content") or ""
        if not isinstance(texto, str):
            texto = json.dumps(texto, ensure_ascii=False)
        uso = datos.get("usage") or {}
        return {
            "texto": texto,
            "tokens_generados": uso.get("completion_tokens") or 0,
            # Aqui no hay duracion de evaluacion: hay que medirla por fuera.
            "tok_por_s": None,
            "tokens_prompt": uso.get("prompt_tokens") or 0,
        }


# --------------------------------------------------------------------------
# Analisis de las respuestas
# --------------------------------------------------------------------------

def contar_palabras(texto: str) -> int:
    return len(re.findall(r"\S+", texto))


def comprobar_racionamiento(texto: str) -> dict:
    """El prompt de calculo tiene respuesta unica y verificable.

    80 * 0,80 = 64 EUR. 120 * 0,65 = 78 EUR. Ahorro = 16 + 42 = 58 EUR.
    """
    esperado = {"A": 64.0, "B": 78.0, "ahorro": 58.0}
    numeros = [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", texto)]

    encontrado = {}
    for clave, valor in esperado.items():
        # Se acepta el valor exacto o su parte entera redondeada.
        if any(abs(n - valor) < 0.01 or abs(round(n) - valor) < 0.01 for n in numeros):
            encontrado[clave] = True
        else:
            encontrado[clave] = False

    return {
        "valores_esperados": esperado,
        "encontrado": encontrado,
        "aciertos": sum(encontrado.values()),
        "total": len(esperado),
    }


def comprobar_codigo(texto: str) -> dict:
    """El prompt de codigo tiene una forma minima verificable.

    No se evalua si el algoritmo es correcto (eso es trabajo de un modelo o de
    una persona), sino que la respuesta sea codigo Python y no una prosa.
    """
    tiene_bloque = "```" in texto
    limpio = re.sub(r"^```(?:python)?\s*|\s*```$", "", texto.strip(), flags=re.M)
    lineas_utiles = [l for l in limpio.splitlines() if l.strip()]

    return {
        "tiene_bloque_codigo": tiene_bloque,
        "prosa_detectada": bool(
            re.search(r"(aquí tienes|explico|la función|el código|claro,)", limpio, re.I)
        ),
        "lineas": len(lineas_utiles),
        "solo_codigo": not re.search(
            r"(aquí tienes|te explico|en resumen)", limpio, re.I
        ),
        "menciona_max": bool(re.search(r"\bmax\b|mayor|highest", limpio)),
    }


def comprobar_instruccion(texto: str) -> dict:
    frases = [f for f in re.split(r"\n+", texto.strip()) if f.strip()]
    numeradas = [f for f in frases if re.match(r"^\s*(\d+[.)]|[-*])\s+", f)]
    cuerpo = [re.sub(r"^\s*\d+[.)]\s*", "", f).strip() for f in numeradas] or frases

    return {
        "lineas": len(frases),
        "numeradas": len(numeradas) == len(frases) and len(frases) > 0,
        "tres_frases": len(frases) == 3,
        "empieza_por_verbo": bool(
            cuerpo and all(re.match(r"^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+e?r?\b|^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+a\b", c) for c in cuerpo)
        ),
    }


def verificar(prompt_id: str, texto: str) -> dict | None:
    """Verificacion automatica por prompt. None si no hay comprobacion posible."""
    if prompt_id == "razonamiento":
        return comprobar_racionamiento(texto)
    if prompt_id == "codigo":
        return comprobar_codigo(texto)
    if prompt_id == "instruccion":
        return comprobar_instruccion(texto)
    return None


# --------------------------------------------------------------------------
# Presentacion
# --------------------------------------------------------------------------

def separador(ancho: int = 100, char: str = "-") -> str:
    return char * ancho


def listar(cliente) -> int:
    print(f"Modelos disponibles via {cliente.nombre}:")
    modelos = cliente.listar_modelos()
    if not modelos:
        print("  (ninguno)")
        return 1
    print()
    print(f"{'Modelo':<40}{'Params':>8}{'Cuantizacion':>16}{'Tamano':>10}")
    print(separador(74))
    for m in modelos:
        tam = f"{m['tamano_gb']:.2f} GB" if m["tamano_gb"] else "-"
        print(f"{m['id']:<40}{m['parametros']:>8}{m['cuantizacion']:>16}{tam:>10}")
    print()
    print(f"{len(modelos)} modelo(s).")
    return 0


def _truncar(texto: str, limite: int = 320) -> str:
    limpio = texto.strip()
    if len(limpio) <= limite:
        return limpio
    return limpio[:limite] + " ..."


def ejecutar(
    cliente,
    modelos: list[str],
    prompts: list[dict],
    temperatura: float,
    max_tokens: int,
    guardar: str | None,
    salida_json: bool,
) -> int:
    resultados = []
    fallos = []

    for prompt in prompts:
        print()
        print(separador(100, "="))
        print(f"PROMPT [{prompt['id']}] — {prompt['dominio']}")
        print(separador(100, "="))
        print(f"  {prompt['texto'][:200]}{'...' if len(prompt['texto']) > 200 else ''}")
        print()

        for modelo in modelos:
            inicio = time.perf_counter()
            registro: dict = {
                "modelo": modelo,
                "prompt_id": prompt["id"],
                "dominio": prompt["dominio"],
                "temperatura": temperatura,
                "max_tokens": max_tokens,
            }
            try:
                salida = cliente.generar(
                    modelo, prompt["texto"], temperatura, max_tokens
                )
            except ClienteError as exc:
                fallos.append({"modelo": modelo, "prompt_id": prompt["id"], "error": str(exc)})
                print(f"  [{modelo}] ERROR: {exc}")
                continue

            elapsed = time.perf_counter() - inicio
            tok = salida["tokens_generados"] or 0
            registro.update(
                {
                    "respuesta": salida["texto"],
                    "tokens_generados": tok,
                    "tokens_prompt": salida["tokens_prompt"],
                    "palabras": contar_palabras(salida["texto"]),
                    "latencia_s": round(elapsed, 2),
                    "tok_por_s_medido": round(tok / elapsed, 2) if elapsed else None,
                    "tok_por_s_motor": salida["tok_por_s"],
                    "verificacion": verificar(prompt["id"], salida["texto"]),
                }
            )
            resultados.append(registro)

            print(f"  [{modelo}]")
            print(f"    {tok} tokens generados · {registro['palabras']} palabras"
                  f" · {elapsed:.2f} s"
                  f" ({registro['tok_por_s_medido']} tok/s)")
            print(f"    {_truncar(salida['texto'])}")
            ver = registro["verificacion"]
            if ver:
                print(f"    verificación: {json.dumps(ver, ensure_ascii=False)}")
            print()

    print(separador(100, "="))
    print("RESUMEN")
    print(separador(100, "="))

    if resultados:
        print(f"{'Modelo':<32}{'Prompt':<14}{'Tokens':>7}{'Palabras':>9}"
              f"{'Latencia':>10}{'tok/s':>8}")
        print(separador(80))
        for r in resultados:
            print(
                f"{r['modelo']:<32}{r['prompt_id']:<14}"
                f"{r['tokens_generados']:>7}{r['palabras']:>9}"
                f"{r['latencia_s']:>9.2f}s{r['tok_por_s_medido']:>8.1f}"
            )

        print()
        print("Aciertos de la verificacion automatica:")
        print(f"{'Modelo':<32}{'Prompt':<14}{'Resultado'}")
        print(separador(64))
        for r in resultados:
            ver = r["verificacion"]
            if not ver:
                continue
            if "aciertos" in ver:
                res = f"{ver['aciertos']}/{ver['total']} valores correctos"
            elif "tres_frases" in ver:
                res = (
                    f"{ver['lineas']} lineas · numeradas: {ver['numeradas']}"
                    f" · 3 frases: {ver['tres_frases']}"
                )
            else:
                res = (
                    f"bloque de codigo: {ver['tiene_bloque_codigo']}"
                    f" · solo codigo: {ver['solo_codigo']}"
                    f" · {ver['lineas']} lineas"
                )
            print(f"{r['modelo']:<32}{r['prompt_id']:<14}{res}")

    if fallos:
        print()
        print(f"Fallos: {len(fallos)}")
        for f in fallos:
            print(f"  {f['modelo']} / {f['prompt_id']}: {f['error'][:110]}")

    print()
    print("Aviso: esto NO es un benchmark de calidad entre modelos.")
    print("Un prompt por modelo, una sola vez, sin verdad terreno completa y sin")
    print("repeticiones, no permite concluir que un modelo sea mejor que otro.")
    print("Lo que si es medible aqui es el coste: tokens, latencia y throughput.")

    if guardar:
        escribir_informe(guardar, cliente, resultados, fallos, temperatura, max_tokens, prompts)
        print(f"\nInforme: {Path(guardar) / 'comparativa_inferencia.md'}")

    if salida_json:
        print()
        print(json.dumps(resultados, ensure_ascii=False, indent=2))

    return 0 if resultados else 1


def escribir_informe(
    destino: str,
    cliente,
    resultados: list[dict],
    fallos: list[dict],
    temperatura: float,
    max_tokens: int,
    prompts: list[dict],
) -> None:
    ruta = Path(destino)
    ruta.mkdir(parents=True, exist_ok=True)
    marca = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    modelos = sorted({r["modelo"] for r in resultados})
    lineas = [
        f"# Comparativa de inferencia — Día 9 ({marca})",
        "",
        f"Backend: **{cliente.nombre}**. Temperatura {temperatura}, "
        f"máximo {max_tokens} tokens.",
        "",
    ]

    if not resultados:
        lineas += ["Sin resultados. Ningún modelo respondió.", ""]
        if fallos:
            lineas += ["## Errores", ""]
            for f in fallos:
                lineas.append(f"- `{f['modelo']}` / {f['prompt_id']}: {f['error']}")
            lineas.append("")

    if resultados:
        lineas += [
            "## Métricas",
            "",
            "| Modelo | Prompt | Tokens gen. | Palabras | Latencia | tok/s |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in resultados:
            lineas.append(
                f"| `{r['modelo']}` | {r['prompt_id']} | {r['tokens_generados']} | "
                f"{r['palabras']} | {r['latencia_s']} s | {r['tok_por_s_medido']} |"
            )

        lineas += ["", "## Verificación automática", ""]
        lineas += [
            "| Modelo | Prompt | Comprobación | Resultado |",
            "|---|---|---|---|",
        ]
        for r in resultados:
            ver = r["verificacion"]
            if not ver:
                continue
            if "aciertos" in ver:
                detalle = f"{ver['aciertos']}/{ver['total']} valores (64, 78, 58 EUR)"
                res = f"{'correcto' if ver['aciertos'] == ver['total'] else 'parcial'}"
            elif "tres_frases" in ver:
                detalle = (
                    f"{ver['lineas']} líneas · numeradas={ver['numeradas']}"
                    f" · exactamente 3={ver['tres_frases']}"
                )
                res = "cumple" if ver["tres_frases"] and ver["numeradas"] else "no cumple"
            else:
                detalle = (
                    f"bloque={ver['tiene_bloque_codigo']} · solo código={ver['solo_codigo']}"
                    f" · {ver['lineas']} líneas"
                )
                res = "cumple" if ver["solo_codigo"] else "no cumple"
            lineas.append(f"| `{r['modelo']}` | {r['prompt_id']} | {detalle} | {res} |")

        lineas += ["", "## Respuestas completas", ""]
        for r in resultados:
            lineas += [
                f"### {r['prompt_id']} — {r['modelo']}",
                "",
                "```text",
                r["respuesta"].strip() or "(vacia)",
                "```",
                "",
            ]

    if fallos:
        lineas += ["## Errores", ""]
        for f in fallos:
            lineas.append(f"- `{f['modelo']}` / {f['prompt_id']}: {f['error']}")
        lineas.append("")

    lineas += [
        "## Qué se puede y qué no se puede concluir",
        "",
        "**Sí:** el coste de la respuesta. Tokens generados, latencia y throughput",
        "son medidas directas y repetibles. En esta máquina el modelo corre sobre CPU,",
        "así que la cifra de tok/s es la de un equipo sin GPU dedicada.",
        "",
        "**No:** que un modelo sea mejor que otro. Un prompt por modelo, una sola",
        "ejecución, sin verdad terreno anotada y sin repeticiones no es un",
        "benchmark. La verificación automática comprueba la *forma* de la respuesta",
        "(3 números correctos, que sea código y no prosa, que haya 3 frases), no su",
        "corrección semántica.",
        "",
        "**La comparación entre modelos del enunciado (Azure) no se ha hecho.**",
        "No hay modelos desplegados a los que acceder. Con más de un modelo local",
        "el script ya produciría la tabla comparativa; con un endpoint compatible",
        "con OpenAI (`--endpoint`), produciría exactamente el entregable original.",
        "",
    ]

    (ruta / "comparativa_inferencia.md").write_text(
        "\n".join(lineas), encoding="utf-8"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def construir_cliente(args) -> tuple[object | None, str | None]:
    if args.endpoint:
        return (
            ClienteOpenAICompatible(args.endpoint, args.api_key or "", args.modelo_unico or ""),
            None,
        )
    if args.host:
        return ClienteOllama(args.host), None
    return ClienteOllama(), None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Inferencia local y comparacion de un mismo prompt (Dia 9)."
    )
    ap.add_argument(
        "--ollama",
        metavar="HOST",
        nargs="?",
        const="http://localhost:11434",
        help="Host de Ollama (por defecto http://localhost:11434).",
    )
    ap.add_argument("--host", metavar="HOST", help="Alias de --ollama.")
    ap.add_argument(
        "--endpoint",
        metavar="URL",
        help="Endpoint compatible con OpenAI (vLLM, llama.cpp server, Azure).",
    )
    ap.add_argument("--api-key", metavar="CLAVE", help="Bearer token del endpoint.")
    ap.add_argument("--modelo-unico", metavar="ID", help="Forzar un id de modelo.")
    ap.add_argument("--listar", action="store_true", help="Solo listar modelos.")
    ap.add_argument(
        "--modelo",
        metavar="M[,M]",
        help="Modelo(s) a comparar, separados por coma.",
    )
    ap.add_argument("--comparar", action="store_true", help="Ejecutar todos los prompts.")
    ap.add_argument("--prompt", metavar="TEXTO", help="Prompt propio en vez de los 3.")
    ap.add_argument("--temperatura", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--guardar", metavar="DIR", help="Escribir informe Markdown en DIR.")
    ap.add_argument("--json", dest="salida_json", action="store_true")
    args = ap.parse_args()

    cliente, error = construir_cliente(args)
    if cliente is None:
        print(error or "No se pudo construir el cliente.", file=__import__("sys").stderr)
        return 2

    print(f"Backend: {cliente.nombre}")
    if not cliente.disponible():
        print()
        print("El backend no responde.")
        if args.endpoint:
            print(f"  URL probada: {cliente.base}")
            print("  Comprueba la URL, el firewall y que --api-key sea correcto.")
        else:
            print("  Ollama deberia estar en http://localhost:11434.")
            print("  Arranquelo con:  ollama serve")
            print("  O instala https://ollama.com")
        return 2

    if args.listar:
        return listar(cliente)

    if args.modelo:
        modelos = [m.strip() for m in args.modelo.split(",") if m.strip()]
    elif isinstance(cliente, ClienteOpenAICompatible) and cliente.modelo_por_defecto:
        modelos = [cliente.modelo_por_defecto]
    else:
        disponibles = cliente.listar_modelos()
        modelos = [m["id"] for m in disponibles]
        if not modelos:
            print("No hay modelos locales. Descarga uno con:  ollama pull llama3.1:8b")
            return 1

    prompts = (
        [{"id": "personalizado", "dominio": "prompt del usuario", "texto": args.prompt}]
        if args.prompt
        else PROMPTS_EJEMPLO
    )

    if not args.comparar and not args.prompt:
        prompts = prompts[:1]

    return ejecutar(
        cliente,
        modelos,
        prompts,
        args.temperatura,
        args.max_tokens,
        args.guardar,
        args.salida_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
