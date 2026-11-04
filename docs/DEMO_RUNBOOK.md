# Hackathon demo runbook

This path is designed for a five-minute, repeatable demo. It uses synthetic imaging data and the deterministic
geometry engine. Muse Spark interprets intent and chooses registered operations; it does not calculate geometry,
invent measurements, or inspect image pixels.

## Before presenting

```bash
./.venv/bin/python tools/demo_preflight.py --port 8000 --require-muse
tools/run_demo.sh 8000
```

Open <http://127.0.0.1:8000/workspace>. Click **Muse**, then **Test connection**. The badge should read
**Muse Spark · connected**. If Meta is unavailable, the product continues with a prominently labelled local
deterministic fallback; do not describe that path as a model response.

Optional KneeScan segmentation is separate. It requires the local service and its Hugging Face model packages.
The synthetic MRI, MPR, annotations, QC, measurements, scaffold engine, and Muse design controls do not depend on it.

## Five-minute story

1. Click **Reset demo**. Introduce the four-pane MPR + Scaffold workspace and the explicit **unregistered** label.
2. Move the linked crosshair in one MRI plane. Show the same physical RAS position in all three planes and mention
   that the phantom is anisotropic, so physical proportions—not square voxel assumptions—drive the display.
3. Open the imaging inspector. Demonstrate window/level, overlay opacity, label colour, and QC as separate controls.
4. Open Muse and use: “Set the sheath wall to 0.3 mm and keep patient geometry unchanged.” Point out the staged
   progress, before/after value, source, validation result, timing, and single **Undo** action.
5. Ask: “Inspect the current design and explain any failed fit checks.” State inspection reads the fresh engine
   revision. A corrective request can only use engine-verified fixes; ambiguity produces a clarification.
6. Show the Evidence → Scaffold view. Emphasize the source annotation, units, target parameter, provenance, and
   assumption status. End with **Undo** or **Reset demo** to prove reversibility.

## Recovery

- **API key required**: put the key in `.env.muse` as `MODEL_API_KEY=...`, restart the server, and use **Test connection**.
- **Authentication failed**: replace the server-side key. Never paste it into the browser or commit it.
- **Model unavailable / rate limited / timed out**: the status names the failure and the request uses the labelled
  fallback when it can be interpreted safely.
- **KneeScan unreachable**: continue with the synthetic case, or start the optional reader at `KNEESCAN_URL`.
- **Unexpected demo state**: click **Reset demo**; it does not delete imported case data.
- **Port busy**: choose another port, for example `tools/run_demo.sh 8123`.

## Claims to avoid

- Do not call model confidence demonstrated accuracy or QC flags medical correctness.
- Do not claim the scaffold is registered to MRI; the current comparison is side by side.
- Do not infer microscopic channels, materials, braid angles, or pretension from routine MRI.
- Do not describe optional 2D KneeScan outlines as validated 3D segmentation.
