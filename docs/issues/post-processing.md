# Post-processing

The scene is shaded into an 8-bit sRGB target and composed straight to the
backbuffer, so highlights clip before anything can roll them off. The phase
begins with an HDR scene target and a tonemap at composition; bloom, TAA, DoF
and SSR follow only after that base is measured.

## P1 - the scene has no headroom and no exposure

What exists: `model.wgsl` applies a per-fragment `1 - exp(-lit)` curve, then
writes into `Rgba8Srgb`. Any value above 1.0 is clipped by the target before
composition, and there is no exposure control at all.

The change:

- The scene colour target becomes `Rgba16Float` and holds linear light. The
  model shader writes `ambient + direct + emissive` with no curve of its own.
- Composition owns the display transform: exposure, an ACES-style curve, and
  the sRGB encode the `Bgra8Srgb` backbuffer applies.
- The editor's grid and gizmos are world-space drawing and stay in the scene,
  so they are tonemapped with it. The 2D sprites and text move after
  composition and draw into the backbuffer: exposure must not dim a crosshair,
  and their authored colours already mean what an sRGB target does.
- `Bifrost::set_exposure` is the control. It is renderer state, not scene data,
  because two viewers of one scene may want different exposures.

**Fixed when:** a test renders the same lit scene at two exposures and the
brighter exposure has higher mean luminance, and the parity/corner tests still
pass on both backends.

## P2 - bloom

Not started. It needs the HDR target from P1, a downsample chain and an additive
upsample before composition. Do not build it on the clipped target.

## P2 - TAA, DoF, SSR

Not started. Each needs a history or depth prepass the current graph does not
declare; decide them after bloom, one at a time.
