# Reconstruction camera rig

Visual hulls (Laurentini 1994, Matusik et al. SIGGRAPH 2000) are the intersection of silhouette cones. Each extra calibrated exterior still carves concavity that a single view cannot see. Generate the stills to match this rig; do not feed cutaways, labels, or diagrams into the hull.

## Locked specimen

One canonical exterior still is the identity lock. Every other view is an `image_edit` of that still (same organ, scale, colour, lighting). Fresh `image_gen` per angle drifts and breaks the hull.

Shared constraints for every request:

- Intact adult human heart, epicardial fat, surface coronaries, great-vessel roots, pointed apex
- Photoreal specimen photography, even studio lighting
- Solid dark grey seamless background (silhouette extraction)
- Organ filling most of a square frame
- No labels, no cutaway, no dissection, no transparency, no multi-panel sheet

## Views (Y-up, +Z anterior, +X patient's left)

| Filename token | Camera | What the silhouette must show |
|---|---|---|
| `anterior` | looking −Z | Sternocostal surface, RV facing camera, apex down-left |
| `posterior` | looking +Z | LA box, pulmonary-vein stumps, IVC |
| `left-lateral` | looking −X | LV left border, left auricle |
| `right-lateral` | looking +X | RA, auricle, SVC/IVC stack |
| `superior` | looking −Y | Base: aorta, pulmonary trunk, SVC, pulmonary veins |
| `inferior` | looking +Y | Diaphragmatic surface, apex toward the bottom of frame |
| `three-quarter` | ~35° left of anterior, slightly above | Combined left-anterior silhouette |

Optional extra angles (tighter hull): three-quarter posterior, high anterior, low anterior.

Inspection stills (coronaries, fat grooves, four-chamber cuts) stay out of `--dir` for reconstruct.

## Prompt pattern

Keep the subject sentence identical. Change only the camera sentence.

```
The same intact adult human heart as the reference, same size and colour, epicardial fat and coronary vessels on the surface, aorta and pulmonary trunk at the base, pointed apex. Photoreal medical specimen on a solid dark grey seamless studio background, even lighting, organ filling the square frame. Camera: <view sentence>.
```
