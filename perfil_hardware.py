"""Perfilado del hardware local y recomendacion de modelo.

Dia 9 - Ecosistema Open Source y Modelos Locales.

Responde a la pregunta util del dia: dado el hardware que tengo delante,
que modelos de la tabla comparativa caben de verdad, y cual elijo.

Los datos de los modelos se leen de modelos_open_source.json. Las cifras de
VRAM de ese fichero son solo pesos: la KV cache se anade aparte y se estima
segun la longitud de contexto que se pida.

Uso:
    python perfil_hardware.py
    python perfil_hardware.py --contexto 32768
    python perfil_hardware.py --json
    python perfil_hardware.py --guardar resultados/
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "modelos_open_source.json"

# Memoria que no queda disponible para pesos: drivers, display, escritorio,
# buffers de contexto. Es una regla practica, no una constante medida.
RESERVA_SISTEMA_GB = 2.0

# bits por peso -> bytes por peso, segun la documentacion de llama.cpp.
BITS_A_BYTES = {
    2.625: 0.328,
    3.91: 0.489,
    4.55: 0.569,
    4.89: 0.611,
    5.70: 0.713,
    6.56: 0.820,
    8.51: 1.064,
}

# De mayor a menor calidad. El primero que cabe es el que se elige.
PRESETS_ORDEN = [
    ("Q8_0", 8.51),
    ("Q6_K", 6.56),
    ("Q5_K_M", 5.70),
    ("Q4_K_M", 4.89),
    ("Q4_0", 4.55),
    ("Q3_K_M", 3.91),
    ("Q2_K", 2.625),
]

# Suelo de calidad. Por debajo de Q4_K_M la degradacion es visible, asi que
# un modelo que solo cabe ahi se marca como 'no recomendable' en vez de
# recomendarlo: caber no es lo mismo que servir bien.
PRESET_MINIMO_UTIL = "Q4_K_M"
PRESETS_DEGRADADOS = {"Q4_0", "Q3_K_M", "Q2_K"}


# --------------------------------------------------------------------------
# Deteccion de hardware
# --------------------------------------------------------------------------

def _memoria_windows() -> tuple[float, float]:
    """(RAM total, RAM libre) en GB, usando GlobalMemoryStatusEx."""
    import ctypes

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = _MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        raise OSError("GlobalMemoryStatusEx fallo")
    return stat.ullTotalPhys / 1024**3, stat.ullAvailPhys / 1024**3


def _memoria_proc() -> tuple[float, float]:
    """(RAM total, RAM disponible) en GB, leyendo /proc/meminfo."""
    total = disponible = 0.0
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for linea in fh:
            if linea.startswith("MemTotal:"):
                total = int(linea.split()[1]) / 1024**2
            elif linea.startswith("MemAvailable:"):
                disponible = int(linea.split()[1]) / 1024**2
                break
    return total, disponible


def _memoria_macos() -> tuple[float, float]:
    """(RAM total, RAM libre aproximada) en GB, en macOS."""
    out = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=15
    )
    total = int(out.stdout.strip()) / 1024**3 if out.stdout.strip().isdigit() else 0.0
    return total, total * 0.6


def ram_gb() -> tuple[float, float]:
    """RAM total y RAM libre del sistema, en GB. (0.0, 0.0) si no se puede leer."""
    try:
        if sys.platform == "win32":
            return _memoria_windows()
        if sys.platform == "darwin":
            return _memoria_macos()
        return _memoria_proc()
    except Exception:
        return 0.0, 0.0


def nombre_cpu() -> str:
    """Nombre del procesador, por la via mas simple que funcione."""
    if sys.platform == "win32":
        for var in ("PROCESSOR_IDENTIFIER", "PROCESSOR_ARCHITECTURE"):
            valor = os.environ.get(var)
            if valor:
                return valor.strip()
        return platform.processor() or "CPU desconocida"

    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for linea in fh:
                if linea.lower().startswith("model name"):
                    return linea.split(":", 1)[1].strip()
    except OSError:
        pass

    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            pass

    return platform.processor() or "CPU desconocida"


def numero_nucleos() -> int:
    nucleos = os.cpu_count() or 1
    if sys.platform.startswith("linux"):
        try:
            nucleos = len(os.sched_getaffinity(0))
        except AttributeError:
            pass
    return nucleos


def _gpus_nvidia() -> list[dict]:
    gpus: list[dict] = []
    if not shutil.which("nvidia-smi"):
        return gpus
    try:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return gpus

    if salida.returncode != 0:
        return gpus

    for linea in salida.stdout.strip().splitlines():
        partes = [p.strip() for p in linea.split(",")]
        if len(partes) >= 2 and partes[1].isdigit():
            gpus.append(
                {
                    "nombre": partes[0],
                    "vram_gb": round(int(partes[1]) / 1024, 2),
                    "fuente": "nvidia-smi",
                }
            )
    return gpus


def _gpus_apple() -> list[dict]:
    try:
        salida = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    for coincidencia in re.finditer(r"(\d+)\s*GB", salida.stdout):
        return [
            {
                "nombre": "GPU integrada (Apple Silicon: memoria unificada)",
                "vram_gb": round(int(coincidencia.group(1)) / 1024, 2),
                "fuente": "system_profiler",
            }
        ]
    return []


def _gpus_amd() -> list[dict]:
    if not shutil.which("rocm-smi"):
        return []
    try:
        salida = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    gpus = []
    for linea in salida.stdout.strip().splitlines()[1:]:
        partes = [p.strip() for p in linea.split(",")]
        if len(partes) >= 2:
            try:
                gpus.append(
                    {
                        "nombre": "GPU AMD",
                        "vram_gb": round(float(partes[1]) / 1024**3, 2),
                        "fuente": "rocm-smi",
                    }
                )
            except ValueError:
                continue
    return gpus


def detectar_gpus() -> list[dict]:
    """GPUs con su VRAM, por el primer metodo que funcione en cada sistema."""
    for detector in (_gpus_nvidia, _gpus_apple, _gpus_amd):
        gpus = detector()
        if gpus:
            return gpus
    return []


def version_ollama() -> str:
    if not shutil.which("ollama"):
        return ""
    try:
        out = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=15
        )
        return out.stdout.strip() or "instalado (version no legible)"
    except (subprocess.SubprocessError, OSError):
        return "instalado (version no legible)"


# --------------------------------------------------------------------------
# Calculo de VRAM
# --------------------------------------------------------------------------

def kv_cache_gb(parametros_b: float, contexto_tokens: int) -> float:
    """KV cache estimada, en GB.

    La KV cache crece linealmente con los tokens, pero el coeficiente depende
    de la arquitectura: capas, KV heads y head dim, y de si el modelo comparte
    KV entre capas (Gemma 4 comparte, asi que paga menos).

    Se usa como referencia la cifra publicada por Meta para Llama 3.1 denso con
    GQA de 8 KV heads: 0,125 GB por cada 1k tokens en 8 B. Se escala de forma
    lineal con los parametros y con un suelo, porque los modelos pequenos
    tienen un coste fijo por token proporcionalmente mayor.
    """
    if contexto_tokens <= 0:
        return 0.0
    por_mil = max(0.125 * max(parametros_b, 1.0) / 8.0, 0.040)
    return por_mil * (contexto_tokens / 1000.0)


def vram_pesos_gb(parametros: int, bits: float) -> float:
    """Memoria de los pesos, en GB. No incluye KV cache ni overhead del motor."""
    if bits not in BITS_A_BYTES:
        bits = min(BITS_A_BYTES, key=lambda b: abs(b - bits))
    return parametros * BITS_A_BYTES[bits] / 1024**3


def cuantizacion_que_cabe(
    parametros: int, vram_util_gb: float, contexto_tokens: int
) -> tuple[str | None, float, float]:
    """Preset de mayor calidad que cabe, con su peso y su total.

    Devuelve (preset, vram_pesos, vram_pesos_mas_kv). Si nada cabe, preset
    es None y los dos numeros valen 0.
    """
    kv = kv_cache_gb(parametros / 1e9, contexto_tokens)
    elegido = (None, 0.0, 0.0)
    for nombre, bits in PRESETS_ORDEN:
        pesos = vram_pesos_gb(parametros, bits)
        if pesos + kv <= vram_util_gb:
            elegido = (nombre, pesos, pesos + kv)
    return elegido


def cabe_con_calidad(
    parametros: int, vram_util_gb: float, contexto_tokens: int
) -> tuple[str | None, float, float]:
    """Igual que cuantizacion_que_cabe, pero sin bajar del suelo de calidad.

    El suelo Q4_K_M SI se evalua: es el preset mas bajo que se recomienda. Solo
    se descartan los que estan por debajo. Devuelve (None, 0, 0) si el modelo
    no entra ni en el suelo.
    """
    kv = kv_cache_gb(parametros / 1e9, contexto_tokens)
    elegido = (None, 0.0, 0.0)
    for nombre, bits in PRESETS_ORDEN:
        if nombre in PRESETS_DEGRADADOS:
            continue
        pesos = vram_pesos_gb(parametros, bits)
        if pesos + kv <= vram_util_gb:
            elegido = (nombre, pesos, pesos + kv)
    return elegido


def cargar_modelos() -> list[dict]:
    with DATOS.open(encoding="utf-8") as fh:
        return json.load(fh)["modelos"]


def evaluar(modelos: list[dict], vram_util_gb: float, contexto_tokens: int) -> list[dict]:
    """Para cada modelo: que preset cabe, a que coste y con cuanto sobra."""
    filas = []
    for m in modelos:
        parametros = m["params_total"]
        preset, pesos, total = cuantizacion_que_cabe(
            parametros, vram_util_gb, contexto_tokens
        )
        preset_ok, pesos_ok, total_ok = cabe_con_calidad(
            parametros, vram_util_gb, contexto_tokens
        )
        # En Gemma 4 el proyector multimodal va en fichero aparte del LLM.
        mmproj = float(m.get("gguf_reales_gb", {}).get("mmproj") or 0.0)
        cabe = preset is not None
        degradado = cabe and preset in PRESETS_DEGRADADOS
        filas.append(
            {
                "id": m["id"],
                "familia": m["familia"],
                "params_legible": m["params_total_legible"],
                "params_efectivos": m.get("params_efectivos"),
                "contexto": m["contexto"],
                "licencia": m["licencia"],
                "cabe": cabe,
                "preset": preset,
                "vram_pesos_gb": round(pesos, 2) if cabe else None,
                "kv_cache_gb": round(
                    kv_cache_gb(parametros / 1e9, contexto_tokens), 2
                ),
                "mmproj_gb": mmproj,
                "vram_total_gb": (round(total + mmproj, 2) if cabe else None),
                "sobra_gb": (round(vram_util_gb - total - mmproj, 2) if cabe else None),
                "recomendable": preset_ok is not None,
                "preset_recomendado": preset_ok,
                "vram_total_recomendado_gb": (round(total_ok + mmproj, 2) if preset_ok else None),
                "degradado": degradado,
            }
        )

    # Primero los que sirven bien, luego los que caben degradados, y dentro de
    # cada grupo de mayor a menor capacidad.
    filas.sort(
        key=lambda f: (f["recomendable"], f["cabe"], f["params_efectivos"] or 0),
        reverse=True,
    )
    return filas


# --------------------------------------------------------------------------
# Presentacion
# --------------------------------------------------------------------------

def separador(ancho: int = 100, char: str = "-") -> str:
    return char * ancho


def miles(n: int) -> str:
    """1234567 -> '1.234.567', para leer los tokens de contexto de un vistazo."""
    return f"{n:,}".replace(",", ".")


def mostrar_hardware(hw: dict) -> None:
    print(separador())
    print("HARDWARE DETECTADO")
    print(separador())
    print(f"  Sistema        {hw['sistema']}")
    print(f"  Python         {hw['python']}")
    print(f"  CPU            {hw['cpu']}")
    print(f"  Nucleos        {hw['nucleos']}")
    print(f"  RAM total      {hw['ram_gb']:.1f} GB")
    print(f"  RAM libre      {hw['ram_libre_gb']:.1f} GB")
    if hw["gpus"]:
        for g in hw["gpus"]:
            print(
                f"  GPU            {g['nombre']} - {g['vram_gb']:.1f} GB VRAM"
                f" ({g['fuente']})"
            )
    else:
        print("  GPU            No detectada (o sin herramienta de consulta)")
    if hw["ollama_version"]:
        print(f"  Ollama         {hw['ollama_version']}")
    else:
        print("  Ollama         No instalado")


def _nombre_corto(modelo_id: str) -> str:
    """'meta-llama/Llama-3.1-8B-Instruct' -> 'Llama-3.1-8B-Instruct'.

    Los identificadores completos no caben en una tabla legible, y la parte
    que distingue al modelo ya es la del nombre, no la del Fabricante.
    """
    return modelo_id.split("/")[-1]


def _celda_preset(fila: dict) -> str:
    """Preset a calidad util, o degradado en parentesis si solo cabe por debajo.

    La columna muestra el preset del suelo de calidad, no el mayor que cabe.
    Si se mostrara el mayor, un modelo de 24 B a Q2_K pareceria mejor que uno
    de 12 B a Q5_K_M, que es justo la confusion que hay que evitar.
    """
    if fila["recomendable"]:
        return fila["preset_recomendado"]
    if fila["cabe"]:
        return f"({fila['preset']})"
    return "-"


def _celda_pesos(fila: dict) -> str:
    if fila["recomendable"]:
        return f"{fila['vram_total_recomendado_gb']:.2f}"
    if fila["cabe"]:
        return f"{fila['vram_total_gb']:.2f}"
    return "-"


def _celda_cabe(fila: dict) -> str:
    if not fila["cabe"]:
        return "no cabe"
    if fila["recomendable"]:
        return "si"
    return "solo degradado"


def mostrar_recomendacion(vram_util: float, contexto: int, filas: list[dict]) -> None:
    print()
    print(separador())
    print(
        f"ESPACIO UTILIZABLE: {vram_util:.1f} GB"
        f"   (techo detectado - {RESERVA_SISTEMA_GB:.1f} GB de reserva del sistema)"
    )
    print(f"CONTEXTO PEDIDO:    {miles(contexto)} tokens")
    print(f"SUELO DE CALIDAD:   {PRESET_MINIMO_UTIL} (por debajo la perdida es visible)")
    print(separador())

    kv_ref = kv_cache_gb(8.03, contexto)
    print()
    print(f"Aviso: la KV cache de {miles(contexto)} tokens ronda {kv_ref:.2f} GB en un")
    print("       modelo de 8 B. Suma a los pesos y no aparece en las tablas de VRAM")
    print("       del fabricante, que solo miden los pesos.")
    print()
    print(
        f"{'Modelo':<34}{'Params':>8}{'Preset':>11}{'KV':>7}"
        f"{'Total':>9}  {'Veredicto'}"
    )
    print(separador(84))

    for f in filas:
        print(
            f"{_nombre_corto(f['id']):<34}{f['params_legible']:>8}"
            f"{_celda_preset(f):>11}{f['kv_cache_gb']:>6.2f}G"
            f"{_celda_pesos(f):>9}  {_celda_cabe(f)}"
        )

    print()
    print("  Total = pesos + KV cache + mmproj. El preset entre parentesis esta")
    print("  por debajo del suelo: cabe en disco, pero la calidad cae y no se")
    print("  recomienda. Prefiere antes un modelo mas pequeño a Q4_K_M.")

    recomendados = [f for f in filas if f["recomendable"]]
    degradados = [f for f in filas if f["cabe"] and not f["recomendable"]]

    if not recomendados:
        print()
        print(">> Ningun modelo de la tabla entra a Q4_K_M con ese contexto.")
        print("   Orden de palancas, de menos a mas invasivo:")
        print("   1. Bajar el contexto (--contexto 4096): reduce la KV cache.")
        print("   2. Bajar a Q5_K_M solo si el modelo es pequeno.")
        print("   3. Aceptar un modelo de la siguiente familia, mas pequeño.")
        if degradados:
            print(f"   Con lo que hay ahora solo caben a preset degradado, por ejemplo:")
            print(f"      {_nombre_corto(degradados[0]['id'])} a {degradados[0]['preset']}"
                  f" -> {degradados[0]['vram_total_gb']:.2f} GB")
        return

    mejor = recomendados[0]
    print()
    print(f">> Recomendacion (mayor capacidad activa que cabe a calidad util):")
    print(f"   {_nombre_corto(mejor['id'])}")
    print(
        f"   {mejor['params_legible']} parametros | preset {mejor['preset_recomendado']}"
        f" | {mejor['vram_total_recomendado_gb']:.2f} GB de {vram_util:.1f} GB"
        f" | licencia {mejor['licencia']}"
    )

    menor = recomendados[-1]
    if mejor["id"] != menor["id"]:
        print(
            f"   Opcion conservadora: {_nombre_corto(menor['id'])}"
            f" ({menor['params_legible']})"
        )

    if degradados:
        cand = degradados[0]
        print()
        print(
            f"   Descartado: {_nombre_corto(cand['id'])} tiene"
            f" {cand['params_legible']}, pero solo entra a {cand['preset']}."
        )
        print("   Preferir un modelo mas pequeño a Q4_K_M antes que uno mayor a Q2_K:")
        print("   Q2_K esta por debajo de lo que la propia documentacion de llama.cpp")
        print("   marca como 'degradacion visible'.")

    print()
    print("   Dos avisos que no estan en la tabla:")
    print("   - En Gemma 4 el mmproj (imagen/audio) va en fichero aparte del LLM")
    print("     y ya esta contado en la columna Total.")
    print("   - En modo CPU la velocidad la marca el ancho de banda de la RAM, no")
    print("     los nucleos. Un modelo mas grande no es mas rapido por mas grande.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Perfilado de hardware y eleccion de modelo (Dia 9)."
    )
    ap.add_argument(
        "--contexto",
        type=int,
        default=8192,
        help="Tokens de contexto que se van a usar (por defecto 8192).",
    )
    ap.add_argument("--json", action="store_true", help="Salida en JSON.")
    ap.add_argument("--guardar", metavar="DIR", help="Escribe el informe JSON en DIR.")
    args = ap.parse_args()

    gpus = detectar_gpus()
    vram_gb = max((g["vram_gb"] for g in gpus), default=0.0)
    ram_total, ram_libre = ram_gb()

    # Sin GPU dedicada, el techo realista es la RAM del sistema.
    if vram_gb > 0:
        techo, modo = vram_gb, "gpu"
    else:
        techo, modo = ram_total * 0.85, "cpu"

    vram_util = max(techo - RESERVA_SISTEMA_GB, 0.5)

    hw = {
        "sistema": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "cpu": nombre_cpu(),
        "nucleos": numero_nucleos(),
        "ram_gb": round(ram_total, 2),
        "ram_libre_gb": round(ram_libre, 2),
        "gpus": gpus,
        "vram_gb": round(vram_gb, 2),
        "modo": modo,
        "ollama_version": version_ollama(),
    }
    hw["ollama_disponible"] = bool(hw["ollama_version"])

    filas = evaluar(cargar_modelos(), vram_util, args.contexto)

    if args.json:
        print(
            json.dumps(
                {
                    "fecha": "2026-09-14",
                    "contexto_tokens": args.contexto,
                    "vram_utilizable_gb": round(vram_util, 2),
                    "hardware": hw,
                    "modelos": filas,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        mostrar_hardware(hw)
        print()
        mostrar_recomendacion(vram_util, args.contexto, filas)

    if args.guardar:
        destino = Path(args.guardar)
        destino.mkdir(parents=True, exist_ok=True)
        fichero = destino / "perfil_hardware.json"
        with fichero.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "fecha": "2026-09-14",
                    "contexto_tokens": args.contexto,
                    "vram_utilizable_gb": round(vram_util, 2),
                    "hardware": hw,
                    "modelos": filas,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nGuardado: {fichero}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
