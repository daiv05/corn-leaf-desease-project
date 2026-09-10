# Verificación del prototipo móvil — ETAPA 2

Fecha de verificación final: `2026-09-09T17:26:10Z`.

## Componente inspeccionado

- Repositorio: `https://github.com/Edenilson-Molina/maize-doctor-app.git`.
- Commit inspeccionado: `dd048d37dcabd08c53c704d702825e7d672f0f37`.
- Stack declarado y comprobado en `package.json`: React Native 0.86, Expo 57,
  TypeScript 6, WatermelonDB y `react-native-fast-tflite`.
- Modelo embarcado: `efficientnet_lite0`, TFLite con pesos int8 y entrada/salida
  float32, 3 743 824 bytes.
- SHA-256 del modelo: `3a0623cd985a23b954e424e605a2e00c91f8d592e376dfe3149e5820485bc2b3`.
- SHA-256 de `labels.json`: `05927e05457ae4e23359f1e3db3830f4c4b73902a2fe814a783331b7ae293957`.
- SHA-256 de `ood_stats.json`: `8f0270945247dae55e37e7efbdc346464223ee1eb2844fb44e04f263a597e92f`.

El motor real carga el recurso TFLite local, valida el contrato 1x3x224x224,
calcula nueve probabilidades y usa el segundo tensor de características para el
detector OOD por distancia relativa de Mahalanobis. La pantalla de captura admite
cámara o galería, guarda el resultado en la base local y muestra diagnóstico,
confianza, segunda posibilidad cuando la confianza es baja y recomendaciones.

## Pruebas ejecutadas

- `npm ci` falló por un conflicto entre React Native 0.86.0 y
  `@react-native/jest-preset` 0.86.2.
- La instalación fijada se completó con `npm ci --legacy-peer-deps`.
- El entorno disponible usa Node 18.19.1, aunque varias dependencias solicitan
  Node 20.19.4 o superior. Es una deuda de reproducibilidad del prototipo.
- `npm run typecheck`: aprobado.
- `npm test -- --runInBand`: 40 suites y 276 pruebas aprobadas. Jest produjo
  advertencias de `act(...)`, pero terminó con código 0.
- `npm audit` integrado en la instalación reportó 24 vulnerabilidades: 17
  moderadas y 7 altas. No se aplicó `npm audit fix --force` al clon verificado.

## Estado del despliegue

Durante la primera comprobación de la sesión, `https://api.doctormaiz.site/health`
respondió `{"status":"ok"}` y `/app-version` informó la versión Android 1.0.0
(código 6) con una URL de APK en GitHub. En la verificación final del mismo día,
el dominio dejó de resolver por DNS y la URL de descarga sin autenticación devolvió
HTTP 404. Por ello la evidencia demuestra una aplicación funcional e integrada,
pero no permite afirmar disponibilidad pública continua del despliegue. La rúbrica
de ETAPA 2 descuenta esa parte.

La captura incluida en el informe se obtiene de la demo ejecutable del experimento
final: toma una imagen real del holdout, aplica el mismo preprocesamiento y produce
Top-3, confianza, latencia y advertencia. No se presenta como medición Android.
