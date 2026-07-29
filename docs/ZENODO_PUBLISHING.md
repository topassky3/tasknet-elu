# Publicación de TaskNet-ELU v0.1.0 en Zenodo

No publicar hasta completar toda esta lista.

## Procedimiento

1. Revisar y fusionar el pull request de preparación.
2. Crear un borrador en Zenodo y reservar o generar el DOI.
3. Sustituir `DOI pendiente de publicación en Zenodo` por el DOI real en el paper, README, `CITATION.cff` y metadatos.
4. Compilar el paper dos veces y verificar referencias, tablas y figuras.
5. Ejecutar:

   ```bash
   python scripts/00_phase0_check.py
   python scripts/05_verify_release.py --check
   python scripts/05_verify_release.py --write-manifest
   ```

6. Crear el tag inmutable `v0.1.0` sobre el commit validado.
7. Crear la GitHub Release y adjuntar el PDF final cuando corresponda.
8. Cargar o archivar la release en Zenodo.
9. Revisar la vista previa completa del registro.
10. Publicar el depósito.
11. Comprobar que el DOI resuelva y que la cita exportada sea correcta.
12. Cualquier corrección posterior deberá publicarse como una versión nueva; no se debe reescribir silenciosamente la v0.1.0.

## Revisión manual obligatoria

Juan Felipe debe confirmar antes de publicar:

- nombre del autor;
- afiliación declarada;
- título y versión;
- resumen y palabras clave;
- licencias MIT y CC BY 4.0;
- fecha de publicación;
- archivos incluidos;
- DOI definitivo;
- que el documento diga “Preprint. No revisado por pares”;
- que no se declare ORCID, financiación o revisión externa inexistente;
- que las cifras coincidan con los CSV versionados.

## Metadatos actuales

- Autor: Juan Felipe Orozco.
- Afiliación: Investigación independiente — Ingeniería de Telecomunicaciones.
- Versión: 0.1.0.
- Tipo: preprint.
- Idioma: español.
- Acceso: abierto.
- DOI: pendiente.
