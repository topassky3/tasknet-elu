# Auditoría de release — TaskNet-ELU v0.1.0

Fecha: 2026-07-29  
Rama: `release/v0.1.0-zenodo`  
Base: `main`

## Resumen

La versión actual contiene un preprint técnico, scripts de las fases 1–4, tablas, figuras e informes. La contribución está correctamente acotada como protocolo experimental reproducible y no como teoría general nueva.

## ERROR

1. `scripts/00_phase0_check.py` todavía exige `src/tasknet_elu` y módulos que fueron eliminados; por tanto, el verificador histórico ya no representa la estructura actual.
2. El README contiene rutas obsoletas para el archivo LaTeX y el PDF.
3. No se localizó en la rama base un archivo LaTeX con el nombre final `paper/TaskNet-ELU-preprint-v0.1.0.tex`; el PDF sí existe bajo el nombre de release. Esto bloquea afirmar que la release compila desde fuente hasta que se recupere o incorpore el `.tex` real.

## ADVERTENCIA

1. `requirements.txt` no fija versiones históricas. No deben inventarse retrospectivamente.
2. Tres semillas ofrecen evidencia inicial, pero no una validación estadística exhaustiva.
3. Energía y latencia son proxies computacionales, no mediciones físicas de red.
4. El embedding está entrenado para la tarea y representa una cota optimista.
5. El DOI todavía no existe y no debe simularse.

## MEJORA

1. Añadir licencia dual explícita.
2. Añadir `CITATION.cff` y `.zenodo.json`.
3. Incorporar verificación de integridad y manifiesto SHA-256.
4. Documentar reproducción de artefactos frente a repetición completa.
5. Añadir CI liviana sin ejecutar los experimentos completos.

## Decisión de publicación

La publicación en Zenodo debe esperar hasta que:

- se recupere o confirme la fuente LaTeX real;
- el paper compile dos veces sin referencias rotas;
- se ejecute el verificador de release;
- se genere el manifiesto SHA-256 definitivo;
- se revise manualmente la cita y los metadatos.

No se modificaron resultados científicos ni se ampliaron las afirmaciones del paper durante esta preparación.
