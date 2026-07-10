# OncoBridge AI — Dataset de evaluación (versión clinical-only)

Dataset de evaluación para el **TP Final de IA Generativa para Biomedicina**. Esta versión está enfocada exclusivamente en el **Componente 1** del sistema (análisis clínico + RAG + ranking de hipótesis diagnósticas). **No incluye estudios de imagen.**

> **Documento de referencia obligatorio:** `OncoBridge_AI_Assignment.md` — contiene el enunciado completo, el contrato de inputs/outputs, los criterios de evaluación y los requisitos de la entrega. Este README solo describe la estructura del dataset.

---

## Estructura de carpetas

```
dataset_clinical_only/
├── README.md                          ← este archivo
└── dataset/
    ├── index.json                     ← índice de los 110 casos (para exploración rápida)
    ├── oncology_ground_truth_base/    ← 30 entradas GT (base de conocimiento del sistema)
    │   ├── GT-BRCA-001.json
    │   ├── GT-BRCA-002.json
    │   └── ...
    └── clinical_cases/                ← 110 casos clínicos para evaluación
        ├── case_001/
        │   ├── input.json             ← input al Componente 1 (datos del paciente)
        │   └── expected_output.json   ← ground truth contra el que comparar
        ├── case_002/
        └── ...
```

---

## 1. Base de ground truth oncológico (`oncology_ground_truth_base/`)

**30 archivos JSON**, uno por entrada. Esta es la **base de conocimiento que el sistema debe consultar via RAG** para producir hipótesis diagnósticas. Está descrita en detalle en la sección §4.1 del enunciado.

### Cobertura

| Órgano | Malignas | Diferenciales benignos | Total |
|---|---|---|---|
| Mama | 4 (carcinoma ductal, lobulillar, DCIS, inflamatorio) | 4 (fibroadenoma, mastitis, quiste, papiloma) | 8 |
| Pulmón / tórax | 4 (adeno, escamoso, células pequeñas, mesotelioma) | 4 (neumonía, granuloma, TBC, EPID) | 8 |
| Colon / GI | 3 | 2 (colitis ulcerosa, diverticulitis) | 5 |
| Próstata | 1 | 1 (HPB) | 2 |
| Tiroides | 1 | 1 (nódulo benigno) | 2 |
| Linfoma | 2 (no Hodgkin, Hodgkin) | — | 2 |
| Hígado | 1 (HCC) | 1 (hemangioma) | 2 |
| Páncreas | 1 (adenoCA ductal) | — | 1 |
| **Total** | **17** | **13** | **30** |

### Esquema de cada GT

```jsonc
{
  "gt_id": "GT-BRCA-001",                    // ID único, formato GT-<CÓDIGO>-<NÚMERO>
  "icd_10": "C50.4",                          // Código ICD-10 oficial
  "icd_10_description": "Neoplasia maligna...",
  "objective_data": {
    "biomarkers": { ... },                    // dict de biomarcadores con umbrales
    "clinical_findings": [ ... ],             // hallazgos al examen físico
    "risk_factors": [ ... ],                  // factores de riesgo
    "prior_imaging_red_flags": [ ... ]
  },
  "subjective_data": {
    "symptoms": [ ... ],                      // síntomas reportados
    "patient_reported_concerns": [ ... ],     // frases en lenguaje del paciente
    "onset_pattern": "..."                    // patrón de evolución
  },
  "radiologist_guidance": {
    "modality_priority": [ ... ],             // ej: ["mammography", "breast_ultrasound"]
    "views_recommended": [ ... ],
    "imaging_location": {
      "body_region": "...",
      "anatomical_landmarks": "...",
      "bilateral_comparison_required": true,
      "priority_zones": [ ... ],
      "positioning_notes": "..."
    },
    "expected_imaging_findings": "...",
    "meddiffusion_prompt": "...",             // para generar imagen de referencia
    "meddiffusion_negative_prompt": "...",
    "image_generation_notes": "..."
  },
  "base_probability": 0.78,                   // probabilidad a priori
  "urgency_level": "alta",                    // alta | media | baja
  "notes": "...",                             // observación clínica
  "_meta": { "category": "...", "organ": "..." }  // metadatos internos (opcionales)
}
```

> Para el detalle clínico de cada campo, ver el ejemplo extendido en §4.1 del enunciado.

---

## 2. Casos clínicos (`clinical_cases/`)

**110 carpetas** (`case_001/` a `case_110/`), cada una con dos archivos:

### `input.json` — input al Componente 1

```jsonc
{
  "patient_id": "PAT-00101",
  "demographics": {
    "age": 58,
    "sex": "F",
    "family_history": ["breast_cancer", "ovarian_cancer"]
  },
  "current_symptoms": [
    "masa palpable CSE mama izquierda",
    "dolor localizado 3 semanas"
  ],
  "medical_history": [
    { "date": "2021-03", "event": "Biopsia mama derecha — fibroadenoma." },
    { "date": "2023-11", "event": "Mamografía bilateral — BI-RADS 2." }
  ],
  "current_labs": {
    "CA_15_3": 42.1,
    "CEA": 6.2,
    "hemograma": "normal"
  }
}
```

> **Nota:** no hay campo `imaging_study` en esta versión. Si tu sistema lo necesita para C2, generá las imágenes vos mismo a partir del `meddiffusion_prompt` del GT correspondiente, o usá un dataset público externo.

### `expected_output.json` — ground truth para evaluación

```jsonc
{
  "case_id": "case_001",
  "correct_gt_ids": ["GT-BRCA-001"],                  // GT(s) que el sistema DEBE matchear
  "acceptable_secondary_gt_ids": ["GT-FIBROA-001"],   // GTs aceptables como secundarios
  "imaging_needed_ground_truth": true,                // si efectivamente requiere imagen
  "urgency_ground_truth": "alta",                     // alta | media | baja | ninguna
  "specialist_decision": "DERIVAR_A_IMAGEN",          // ver tabla abajo
  "conclusive_ground_truth": true,                    // si hay hipótesis conclusiva
  "difficulty": "facil",                              // facil | moderado | dificil
  "notes": "Carcinoma ductal CSE clásico en alto riesgo."
}
```

**Valores posibles de `specialist_decision`:**

| Valor | Significado |
|---|---|
| `DERIVAR_A_IMAGEN` | El sistema debe recomendar derivación a estudio de imagen |
| `NO_DERIVAR` | El sistema identifica condición benigna o sin necesidad de imagen |
| `SEGUIMIENTO_CLINICO` | Sin urgencia de imagen pero requiere seguimiento |
| `SIN_ELEMENTOS_PARA_EVALUAR` | Datos insuficientes para formular hipótesis |

---

## 3. Composición de los 110 casos

| Categoría | N | Qué evalúa |
|---|---|---|
| **TP** (true positive) | 30 | Cáncer claro distribuido entre los 17 GT oncológicos — debe derivar |
| **TN** (true negative) | 30 | 15 `SIN_ELEMENTOS` (sano) + 15 `NO_DERIVAR` (match benigno claro) |
| **FP** (false positive borderline) | 15 | Síntomas sospechosos pero benigno (mastitis, neumonía, diverticulitis, hemangioma, HPB) — el sistema debe NO sobre-derivar |
| **FN** (false negative sutil) | 15 | Cáncer atípico (lobulillar sin masa, pulmón en no fumador, páncreas con diabetes nueva) — el sistema debe alertar pese a perfil atípico |
| **COMPLEX** | 20 | Historial extenso (8-12 eventos por paciente) — estresa la **eficiencia de contexto** del LLM (criterio §4.0 del enunciado) |

### Por dificultad

- `facil`: 48 casos (43%)
- `moderado`: 40 casos (36%)
- `dificil`: 22 casos (20%)

### Por decisión esperada

- `DERIVAR_A_IMAGEN`: 78 casos
- `NO_DERIVAR`: 16 casos
- `SIN_ELEMENTOS_PARA_EVALUAR`: 15 casos
- `SEGUIMIENTO_CLINICO`: 1 caso

---

## 4. `index.json` — índice de exploración rápida

Resumen tabular de los 110 casos con sus IDs, categoría, GT correcto y dificultad. Útil para:

- Explorar el dataset sin tener que abrir 110 carpetas
- Filtrar casos por categoría o dificultad
- Construir loaders en Python / Pandas

```python
import json
index = json.load(open("dataset/index.json"))
for case in index["cases"]:
    print(case["case_id"], case["category"], case["correct_gt_ids"])
```

---

## 5. Cómo cargar el dataset en Python

```python
import json
from pathlib import Path

ROOT = Path("dataset")

# Cargar todas las entradas GT
gt_base = {}
for gt_file in (ROOT / "oncology_ground_truth_base").glob("*.json"):
    entry = json.loads(gt_file.read_text())
    gt_base[entry["gt_id"]] = entry

# Cargar todos los casos clínicos
cases = []
for case_dir in sorted((ROOT / "clinical_cases").glob("case_*")):
    cases.append({
        "case_id": case_dir.name,
        "input": json.loads((case_dir / "input.json").read_text()),
        "expected": json.loads((case_dir / "expected_output.json").read_text()),
    })

print(f"GT entries: {len(gt_base)}")    # 30
print(f"Cases: {len(cases)}")            # 110
```

---

## 6. Cómo usar este dataset en el TP

1. **Sistema (Componente 1):** indexar la base de ground truth con la estrategia RAG que diseñe el equipo. La base tiene 30 entradas — el sistema debe poder consultarla eficientemente respetando la restricción de contexto descripta en §4.0 del enunciado.

2. **Evaluación:** correr cada uno de los 110 casos contra el sistema, comparando el output producido con el `expected_output.json` correspondiente. El script de evaluación debe medir:

   - **Precisión de GT match:** % de casos donde `correct_gt_ids` está entre los devueltos.
   - **Calibración:** correlación entre `match_probability` reportada y la frecuencia real de acierto.
   - **Accuracy de derivación:** coincidencia con `specialist_decision` y `imaging_needed_ground_truth`.
   - **Eficiencia de tokens:** prompt + completion tokens promedio por caso.

3. **Componente 2 (asistencia radiológica):** este dataset **no incluye estudios de imagen**. Para implementar y evaluar C2, el equipo puede:
   - Generar imágenes sintéticas a partir de los `meddiffusion_prompt` de los GT enlazados.
   - Usar un dataset público externo (CBIS-DDSM, ChestX-ray14, MedMNIST).
   - Diseñar una versión textual del input para C2 si el alcance se restringe a C1.

   La decisión y su justificación es parte del trabajo.

---

## 7. Limitaciones a tener en cuenta

1. **Contenido sintético:** los casos y la base GT están anclados en guías clínicas públicas (NCCN, ESMO, ICD-10 WHO, BI-RADS) pero no fueron revisados por especialistas certificados. **No usar para decisiones clínicas reales.**

2. **Sin pediatría ni embarazo:** el dataset cubre adultos (22-79 años) y no incluye condiciones gestacionales ni pediátricas.

3. **Distribución de GTs no uniforme:** ca colorrectal, mama y pulmón están sobre-representados — refleja la prevalencia clínica real.

4. **Casos de borde intencionales:** algunos casos referencian dominios oncológicos sin GT exacto en la base (ej. `case_020` ca gástrico, `case_109` ca ovario). Son intencionales para evaluar cómo el sistema responde cuando no hay match perfecto.

5. **`base_probability` aproximada:** los valores en los GT son orientativos. Parte del trabajo del equipo es decidir si recalibrarlos según los datos del paciente y documentar su fórmula (ver §4.2 del enunciado).
