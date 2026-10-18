donc# SADT — Résultats expériences

> Généré le 2026-10-05. Jobs exp5/rebuild_1 (SLURM 1456/1457) ont crashé — voir section en bas.

---

## Exp 1 — Replication LLCC (n=500)

### Run fp16 — notre CHAIR + semantic matcher predefined (job 1465, 2026-10-06)

Source : `outputs/exp1_fp16/summary.json`

| Variante | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Paper (LLCC) | 0.820 | 0.800 | **0.810** | 355 | 78 | 89 |
| Baseline (notre CHAIR) | 0.531 | 0.628 | 0.576 | 314 | 277 | 186 |
| Sink-corrected | 0.516 | 0.554 | 0.534 | 277 | 260 | 223 |

- Δ F1 (notre baseline vs paper) : **−0.234**
- Δ F1 (sink-corrected vs baseline) : **−0.042** — la correction sink *détériore* légèrement
- Cause du gap identifiée : semantic matcher `predefined` trop pauvre (1 paire seulement), + captions différentes (53% GT mismatch)

### Run paper-replication — mots papier + COCO matcher (job soumis 2026-10-06)

Source : `outputs/exp1_paper_replication/summary.json` *(en cours)*

---

## Exp 2a — Attention PR vs hallucination (image-level, n=500)

Source : `outputs/exp2_fp16/summary.json`

| Groupe | PR médian | PR moyen |
|---|---|---|
| Hallucinated (n=249) | 0.02643 | 0.02739 |
| Clean (n=251) | 0.02917 | 0.03037 |

- AUC (low PR → hallu) : **0.609** — signal faible mais significatif
- Mann-Whitney p : **2.5×10⁻⁵**
- Leading-ratio AUC : 0.502 (p=0.932) — pas discriminant

---

## Exp 2b — Attention PR par mot (word-level TDO causal, n=1679 mots)

Source : `outputs/exp2_causal_tdo_fp16/summary.json`

| Groupe | n | PR médian | AUC | p |
|---|---|---|---|---|
| Grounded | 1205 | 0.07548 | — | — |
| Type 1 (hallu) | 372 | 0.04052 | 0.215 | 4.7×10⁻⁶² |
| Type 2 (hallu) | 102 | 0.04245 | 0.240 | 2.3×10⁻¹⁸ |

Les mots hallucinés reçoivent significativement moins d'attention visuelle.

---

## Exp 2c — Attention PR par objet (object-level TDO, n=497 images)

Source : `outputs/exp2_objects_tdo_fp16/summary.json`

| Groupe | n | PR médian | AUC | p |
|---|---|---|---|---|
| Grounded | 214 | 0.776 | — | — |
| Type 1 (hallu) | 204 | 0.502 | 0.173 | 5.7×10⁻³¹ |
| Type 2 (hallu) | 79 | 0.399 | 0.053 | 5.5×10⁻³² |

Même signal, encore plus fort à l'échelle objet.

---

## Exp 2d — Variante spectrale (spectral TDO, n=500)

Source : `outputs/exp2_spectral_tdo/summary.json`

| Groupe | PR médian | PR moyen |
|---|---|---|
| Hallucinated (n=249) | 0.02773 | 0.02869 |
| Clean (n=251) | 0.03105 | 0.03207 |

- AUC : **0.609** (identique à 2a)
- Mann-Whitney p : **2.7×10⁻⁵**
- Leading-ratio AUC : 0.513 (p=0.623) — pas discriminant

---

## Exp 3 — Correction par effacement (CHAIR, n=293 images hallucinées)

Source : `outputs/exp3_fp16/summary.json`

### Toutes images hallucinées (n=293)

| Méthode | CHAIRs | CHAIRi |
|---|---|---|
| Original | 1.000 | 0.370 |
| SADT regen | **0.853** | **0.324** |
| Exp3 erase | 0.911 | 0.320 |

### Images Type 2 seulement (n=79)

| Méthode | CHAIRs | CHAIRi |
|---|---|---|
| Original | 1.000 | 0.431 |
| SADT regen | 1.000 | 0.413 |
| Exp3 erase | **0.671** | **0.264** |

L'effacement Type 2 est particulièrement efficace sur les images Type 2.

---

## Exp 4 — Classification Type 1 vs Type 2 par régression logistique (n=367)

Source : `outputs/exp4_fp16/summary.json`

| Métrique | Valeur |
|---|---|
| n samples | 367 (Type1=267, Type2=100) |
| LOO accuracy | 0.728 |
| Baseline accuracy | 0.728 |
| LOO F1 | 0.000 |
| LOO AUC | 0.486 |

Coefficients : entropy=+0.124, concentration=−0.133, sink_ratio=−0.074

Le classifieur n'apporte rien au-dessus de la baseline (majorité = Type 1).

---

## Exp 1 — Relance (job 1458, 2026-10-05)

**Annulé par TIME LIMIT à 416/500.** Aucun nouveau résultat écrit (le script écrit à la fin).

Cause : time limit trop courte (~2h) vs durée réelle ~2h30. Les résultats `exp1_fp16` actuels datent d'un run antérieur incomplet (n=1, tout à zéro).

---

## Exp 5 — Locality vs information selection (SAUQ, n=6 couches)

### Run fp16 (job 1466, 2026-10-06)

Source : `outputs/exp5_locality/summary.json`

| Métrique | Local (mean±std) | Global (mean±std) | Mann-Whitney p | Spearman ρ | Spearman p |
|---|---|---|---|---|---|
| row_sq | 0.0151±0.0007 | 0.0167±0.0009 | **0.015** | 0.724 | 0.008 |
| row_entropy | 0.339±0.038 | 0.631±0A.029 | **0.002** | **0.869** | **0.0002** |
| NLP | 0.029±0.017 | 0.142±0.070 | **0.002** | **0.869** | **0.0002** |
| mean_token_entropy | 0.065±0.036 | 0.314±0.147 | **0.002** | **0.869** | **0.0002** |

Résultat fort : les tokens locaux ont une entropie nettement plus faible que les tokens globaux (p ≤ 0.015 pour toutes les métriques, ρ ≈ 0.87).

---

## Rebuild 1 — Figure 4 copy task replication (SAUQ, job 1474, 2026-10-06)

Source : `outputs/rebuild_1_copy_task/summary.json`

### Valeurs absolues (moyennes sur 3 trials)

| n | NLP α | NLP β | Entropy α | Entropy β | RAUQ α | RAUQ β | SAUQ α | SAUQ β |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.0081 | 0.0122 | 0.0478 | 0.0644 | 2.471 | 2.472 | 1.981 | 2.010 |
| 20 | 0.0270 | 0.0342 | 0.0676 | 0.0984 | 2.534 | 2.553 | 2.248 | 2.232 |
| 30 | 0.0178 | 0.0194 | 0.0666 | 0.0756 | 2.535 | 2.549 | 2.262 | 2.272 |
| 50 | 0.0073 | 0.0111 | 0.0385 | 0.0460 | 2.540 | 2.549 | 2.286 | 2.283 |
| 75 | 0.0032 | 0.0067 | 0.0185 | 0.0273 | 2.548 | 2.556 | 2.307 | 2.297 |
| 100 | 0.0546 | 0.0756 | 0.1340 | 0.1569 | 2.402 | 2.492 | 1.932 | 2.081 |

### Gaps β − α

| n | ΔNLP | ΔEntropy | ΔRAUQ | ΔSAUQ |
|---|---|---|---|---|
| 10 | +0.004 | +0.017 | +0.002 | **+0.030** |
| 20 | +0.007 | +0.031 | +0.019 | −0.016 |
| 30 | +0.002 | +0.009 | +0.014 | +0.010 |
| 50 | +0.004 | +0.007 | +0.009 | −0.003 |
| 75 | +0.004 | +0.009 | +0.007 | −0.010 |
| 100 | +0.021 | +0.023 | **+0.091** | **+0.149** |

### Conclusion

**Ne reproduit pas la Figure 4 du papier.** Le papier prédit un gap RAUQ/SAUQ croissant et un NLP qui converge — on observe ni l'un ni l'autre de façon nette. Causes probables :
- 3 trials seulement → variance trop haute
- LLaVA-7B copie trop bien pour n ≤ 75 (NLP ≈ 0 pour les deux familles), aucune métrique ne peut distinguer
- Seul n=100 montre un vrai gap (le modèle commence à échouer), mais c'est là que NLP diverge aussi — pas spécifique à SAUQ

---

## Exp 1 — Paper-A replication (n=250, COCO matcher)

Source : `outputs/exp1_paper_A/summary.json`

| Variante | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Paper (LLCC) | 0.794 | 0.787 | **0.791** | 185 | 48 | 50 |
| Baseline (notre CHAIR) | 0.592 | 0.438 | 0.504 | 103 | 71 | 132 |
| Sink-corrected | 0.531 | 0.328 | 0.405 | 77 | 68 | 158 |

- Δ F1 (baseline vs paper) : **−0.287**
- Δ F1 (sink-corrected vs baseline) : **−0.098** — la correction sink *détériore* encore
- n_images : 250

---

## Rebuild 1 — Figure 4 full (SAUQ, étendu — n=5 à 100)

Source : `outputs/rebuild_1_full/summary.json`

Extension de `rebuild_1_copy_task` avec n=5 et n=15 ajoutés.

### Valeurs absolues (moyennes α easy / β hard)

| n | NLP α | NLP β | Entropy α | Entropy β | RAUQ α | RAUQ β | SAUQ α | SAUQ β |
|---|---|---|---|---|---|---|---|---|
| 5  | 0.0155 | 0.0258 | 0.0787 | 0.0860 | 2.379 | 2.400 | 1.912 | 1.961 |
| 10 | 0.0081 | 0.0133 | 0.0478 | 0.0634 | 2.471 | 2.465 | 1.981 | 2.015 |
| 15 | 0.0321 | 0.0265 | 0.0686 | 0.0789 | 2.491 | 2.509 | 2.149 | 2.115 |
| 20 | 0.0270 | 0.0306 | 0.0676 | 0.0845 | 2.534 | 2.547 | 2.248 | 2.219 |
| 30 | 0.0178 | 0.0250 | 0.0666 | 0.0823 | 2.535 | 2.555 | 2.262 | 2.271 |
| 50 | 0.0073 | 0.0088 | 0.0385 | 0.0436 | 2.540 | 2.548 | 2.286 | 2.276 |
| 75 | 0.0032 | 0.0087 | 0.0185 | 0.0334 | 2.548 | 2.549 | 2.307 | 2.280 |
| 100| 0.0546 | 0.1059 | 0.1340 | 0.2171 | 2.402 | 2.451 | 1.932 | 1.972 |

### Gaps β − α

| n | ΔNLP | ΔEntropy | ΔRAUQ | ΔSAUQ |
|---|---|---|---|---|
| 5  | +0.010 | +0.007 | +0.021 | +0.049 |
| 10 | +0.005 | +0.016 | −0.006 | +0.034 |
| 15 | −0.006 | +0.010 | +0.018 | −0.034 |
| 20 | +0.004 | +0.017 | +0.013 | −0.030 |
| 30 | +0.007 | +0.016 | +0.020 | +0.009 |
| 50 | +0.002 | +0.005 | +0.008 | −0.010 |
| 75 | +0.005 | +0.015 | +0.001 | −0.026 |
| 100| +0.055 | +0.083 | **+0.049** | **+0.040** |

Pas de tendance claire : les gaps RAUQ/SAUQ ne croissent pas avec n, et alternent en signe. Confirme la non-réplication de rebuild_1_copy_task.

---

## Audit multi-token — LLCC sur mots multi-sous-tokens (n=48 images)

Source : `outputs/multitoken_audit/summary.json`

| Groupe | n mots | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Tous | 148 | 0.606 | 0.690 | **0.645** | 20 | 13 | 9 |
| Mono-token | 65 | 0.611 | 0.786 | **0.688** | 11 | 7 | 3 |
| Multi-token | 83 | 0.600 | 0.600 | **0.600** | 9 | 6 | 6 |

### Performance par nombre de sous-tokens

| n_subtokens | Precision | Recall | F1 | n |
|---|---|---|---|---|
| 2 | 0.556 | 0.625 | 0.588 | 12 |
| 3 | 0.333 | 0.500 | 0.400 | 4 |
| 4 | 1.000 | 0.600 | 0.750 | 5 |

Config : layers=[20..27], semantic_matcher=coco, topk_patches=3

Les mots mono-token sont mieux détectés (F1=0.688 vs 0.600). Les mots à 3 sous-tokens sont les plus difficiles (F1=0.400). L'échantillon est petit (n=48) — à confirmer à plus grande échelle.

---

## Exp 1 — COCO synonyms (n=500, semantic_matcher=coco)

Source : `outputs/exp1_coco_synonyms/summary.json`

| Variante | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Paper (LLCC) | 0.820 | 0.800 | **0.810** | 355 | 78 | 89 |
| Baseline (notre CHAIR) | 0.611 | 0.532 | **0.569** | 266 | 169 | 234 |
| Sink-corrected | 0.598 | 0.438 | **0.506** | 219 | 147 | 281 |

- Δ F1 (baseline vs paper) : **−0.241**
- Δ F1 (sink-corrected vs baseline) : **−0.063** — sink détériore encore
- n_images : 500

Le matcher `coco` améliore légèrement le baseline vs `predefined` (F1=0.504 → 0.569) mais ne ferme pas le gap avec le papier.

---

## Exp 6 — Type-2 masking combiné (n=200)

Source : `outputs/exp6_type2_masking/summary.json`

| Variante | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Paper (LLCC) | 0.794 | 0.782 | **0.788** | 154 | 40 | 43 |
| Baseline (LLCC) | 0.549 | 0.502 | **0.525** | 106 | 87 | 105 |
| Exp6 combined (Type1 ∪ Type2) | 0.255 | 0.768 | **0.383** | 162 | 473 | 49 |

- Δ F1 (baseline vs paper) : **−0.263**
- Δ F1 (exp6 vs baseline) : **−0.142** — exp6 est pire que baseline
- Δ F1 (exp6 vs paper) : **−0.405**
- n_images : 200

Le Type-2 (masquage + régénération) fait exploser les FP (87 → 473) : le détecteur hallucine sur les nouvelles captions régénérées. Recall monte (0.502 → 0.768) mais precision s'effondre (0.549 → 0.255) → F1 net pire.

---

## Exp 7 — Iterative masking (n=80)

Source : `outputs/exp7_iterative_masking/summary.json`

| Variante | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Paper (LLCC) | 0.892 | 0.795 | **0.841** | 58 | 7 | 15 |
| Baseline (LLCC) | 0.613 | 0.469 | **0.531** | 38 | 24 | 43 |
| Exp7 combined (union itérations) | 0.404 | 0.494 | **0.444** | 40 | 59 | 41 |

- Δ F1 (baseline vs paper) : **−0.309**
- Δ F1 (exp7 vs baseline) : **−0.087** — exp7 est pire que baseline
- Δ F1 (exp7 vs paper) : **−0.396**
- avg_n_regen : **1.95** (≈ 2 itérations en moyenne)
- CHAIRs final : **0.625** | CHAIRi final : **0.265**
- n_images : 80

L'union des détections sur plusieurs itérations accumule les FP (24 → 59) sans gain suffisant en TP (38 → 40). La mitigation via masquage (CHAIRs=0.625, CHAIRi=0.265) ne semble pas spectaculaire — à comparer avec un CHAIRs baseline pour juger.
