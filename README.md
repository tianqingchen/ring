# Ring models

Millimetre-accurate Blender ring models and geometry previews.

## Layout

- `models/` — Blender builders and `.blend` files (`pink-prongs`, `blue-integrated`, `blue-grips`)
- `media/geometry/` — clay / workbench views for geometry review
- `media/renders/` — product Cycles renders (pink finals may be stale vs latest geometry)
- `references/` — IGI certs and concept photos
- `preferences.md` — sizing notes (US 5 / 15.7 mm ID)

## Build

Blender 4.5+:

```bash
blender --background --python models/build_rings.py -- --ring pink --stage mesh
```

Rings: `pink`, `integrated`, `grips`, or `all`. Stages: `mesh`, `preview`, `color`, `final`.
