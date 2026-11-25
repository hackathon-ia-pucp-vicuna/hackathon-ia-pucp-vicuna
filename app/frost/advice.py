"""
Mensajes fijos en español por nivel (v0 · T44). T16 (@lalo-dev8) los redacta y valida con un usuario real.
Máximo 3 oraciones: situación, recomendación e incertidumbre.
"""
MENSAJES = {
    "sin riesgo": (
        "Esta noche no se espera helada en tu zona. "
        "No hace falta proteger la papa. "
        "Es un pronóstico: revisa de nuevo mañana por la tarde."
    ),
    "vigilancia": (
        "Esta noche hará mucho frío y podría helar en las partes bajas o abiertas. "
        "Ten listo riego ligero o cobertura por si baja más la temperatura. "
        "El pronóstico puede fallar por 1 o 2 °C."
    ),
    "helada probable": (
        "Esta noche es probable que hiele en tu zona. "
        "Protege la papa antes de la hora de inicio con riego ligero, humo o cobertura. "
        "El pronóstico puede fallar por 1 o 2 °C."
    ),
}

AVISO_LIMITES = (
    "El modelo solo estima temperatura, humedad y viento: no predice lluvia, nubosidad, radiación ni ráfagas."
)


def mensaje(nivel):
    return MENSAJES[nivel]
