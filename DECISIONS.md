# Decisiones de implementación — Backend

Registro de decisiones tomadas al construir el backend, siguiendo el mismo estilo que el repo
de la app.

## Stack y estructura

- **FastAPI + uvicorn** (indicado en `SETUP.md`). Paquete `app/` con `routers/` (un fichero por
  endpoint) y `services/` (LLM, preprocess, cache, rate_limit, revenuecat, oplog). Sin capas de
  abstracción innecesarias, mismo criterio que la app.
- **SQLAlchemy core + PyMySQL** para hablar con MySQL. Síncrono (las llamadas a BD son triviales;
  la latencia la domina el LLM). Sin ORM/migraciones: 3 tablas en `deploy/schema.sql` + creación
  idempotente al arrancar.

## Contrato con la app

- Los modelos Pydantic usan **snake_case** en los nombres de campo para casar exactamente con lo
  que envía/decodifica la app (`APIModels.swift`, que usa convert-to/from-snake-case). Verificado
  campo a campo en los tests.
- `/categorize` recibe **labels** y debe devolver un label copiado **verbatim** de la lista
  permitida. Si el modelo responde con casing distinto, se resuelve al label canónico de la lista;
  si no está en la lista, se devuelve `category: null` (la app lo deja sin categorizar).

## Detección de tipo de input (decisión clave)

- `ParseRequest` lleva `filename`, pero `EnrichRequest` **no**. Para no tocar la app, el backend
  **detecta PDF/Excel/imagen por magic bytes** del contenido base64 (`%PDF`, `PK\x03\x04`, OLE2,
  cabeceras JPEG/PNG/GIF/WEBP), usando la extensión del `filename` solo como desempate. Así
  `/enrich` rutea tickets en PDF aunque no reciba nombre de fichero.
- PDF → texto con **PyMuPDF**; Excel → texto con **openpyxl**; imagen real → data URI al modelo
  multimodal. Los parsers se importan de forma perezosa para que el módulo (y los tests) carguen
  sin esas dependencias.

## Resiliencia / degradación

- Si `DATABASE_URL` está vacío o MySQL no responde: cache, rate limiting y logging operacional se
  convierten en no-ops (**fail open**) y la API sigue sirviendo. El producto es el proxy al LLM;
  la BD es una optimización.
- Router de proveedores: primario (nan.builders) → fallback (OpenAI). Cualquier error HTTP/parse
  del primario salta al siguiente. Se registra quién respondió, si fue fallback y el motivo del
  fallo del primario (SPECS §11.5).

## Premium gate

- `/enrich` valida la suscripción contra la **RevenueCat REST API** con la **secret key** del
  servidor (nunca la pública del iOS). `REQUIRE_PREMIUM=false` desactiva el gate para desarrollo.
  Si el gate está activo pero no se puede verificar (sin key / RevenueCat caído) → **503** (no se
  asume premium). Sin premium → **403** (la app ya maneja 403 mostrando el paywall).

## Cache de categorización

- Clave: `signo + concepto normalizado` (minúsculas, espacios colapsados). Un hit solo se reutiliza
  si el label cacheado sigue estando en la lista de categorías que envía la app, así categorías
  renombradas/eliminadas nunca resucitan un valor obsoleto. TTL configurable (30 días por defecto).
