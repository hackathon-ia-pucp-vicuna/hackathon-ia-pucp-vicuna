"""
Mensajes en español por nivel de riesgo (T16 · HU-03).

Cada mensaje tiene como máximo 3 oraciones: situación, recomendación e incertidumbre.
Las plantillas usan campos {…} que se rellenan con el resultado de rules.evaluar_noche();
si falta un dato, se usa la versión general del mensaje (nunca queda un campo sin rellenar).
Los textos deben revisarse con un usuario real (T28) antes de darlos por definitivos.
"""
import string

INCERTIDUMBRE = "Es un pronóstico y puede fallar por 1 o 2 °C, sobre todo en hondonadas y zonas abiertas."

PLANTILLAS = {
    "sin riesgo": {
        "detalle": "Esta noche no se espera helada en {zona}: la mínima rondaría {tmin} °C.",
        "general": "Esta noche no se espera helada en tu zona.",
        "recomendacion": "No hace falta proteger la papa; revisa el aviso de nuevo mañana por la tarde.",
    },
    "vigilancia": {
        "detalle": "Esta noche hará mucho frío en {zona}, con una mínima cercana a {tmin} °C hacia las {hora_tmin}.",
        "general": "Esta noche hará mucho frío en tu zona y podría helar en las partes bajas.",
        "recomendacion": "Ten listo el riego ligero o la cobertura de la papa por si la temperatura baja más.",
    },
    "helada probable": {
        "detalle": "Es probable que hiele en {zona} desde las {inicio}, durante unas {duracion} h, con mínima de {tmin} °C.",
        "general": "Es probable que hiele esta noche en tu zona.",
        "recomendacion": "Protege la papa antes del anochecer con riego ligero o cubriendo las plantas.",
    },
}

AVISO_LIMITES = (
    "El modelo solo estima temperatura, humedad y viento: no predice lluvia, nubosidad, radiación ni ráfagas."
)


def campos(plantilla):
    """Nombres de los campos {…} de una plantilla."""
    return {c for _, c, _, _ in string.Formatter().parse(plantilla) if c}


def _datos(resultado, zona):
    """Valores para rellenar las plantillas; omite los que no están disponibles."""
    d = {}
    if zona:
        d["zona"] = zona
    if not resultado:
        return d
    if resultado.get("tmin_c") is not None:
        d["tmin"] = str(int(round(resultado["tmin_c"])))      # entero: evita "-0" y decimales con punto
    if resultado.get("hora_tmin") is not None:
        d["hora_tmin"] = f"{resultado['hora_tmin']:%H:%M}"
    if resultado.get("inicio") is not None:
        d["inicio"] = f"{resultado['inicio']:%H:%M}"
    if resultado.get("duracion_h"):
        d["duracion"] = f"{resultado['duracion_h']:g}".replace(".", ",")
    return d


def mensaje(nivel, resultado=None, zona=None):
    """
    Mensaje de 3 oraciones para el nivel ('sin riesgo', 'vigilancia', 'helada probable').
    resultado : dict de rules.evaluar_noche() (opcional) · zona : nombre de la localidad (opcional).
    """
    p = PLANTILLAS[nivel]
    datos = _datos(resultado, zona)
    situacion = p["detalle"].format(**datos) if campos(p["detalle"]) <= set(datos) else p["general"]
    return f"{situacion} {p['recomendacion']} {INCERTIDUMBRE}"
