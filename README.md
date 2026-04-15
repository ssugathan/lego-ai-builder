# lego-ai-builder

Converts a 2D image into a JSON description of LEGO-style 3D primitives.

## Setup

```bash
pip install pydantic
```

## Run

```bash
python pipeline.py examples/input.png
```

Pipe to a file:

```bash
python pipeline.py examples/input.png > model.json
```

## Sample Output

```json
{
  "source_image": "input.png",
  "baseplate": { "width": 50, "depth": 50 },
  "bounding_box": { "x": 12, "y": 0, "z": 23, "width": 4, "height": 3, "depth": 4 },
  "shapes": [
    {
      "id": "shape_0",
      "type": "cuboid",
      "position": { "x": 12, "y": 0, "z": 23 },
      "dimensions": { "width": 4, "height": 3, "depth": 4 },
      "rotation": 0,
      "color": "#FF0000",
      "label": "body"
    }
  ]
}
```

## Pipeline Overview

| Stage | Function | Status |
|---|---|---|
| Interpret image | `interpret_image()` | Stub — returns fixed scene |
| Choose primitive + dimensions | `_make_dimensions()` | Deterministic, driven by `primitive_hint` + `relative_size` |
| Place shapes on baseplate | `decompose_into_primitives()` | Spreads shapes evenly along X, centred on Z |

## Notes

- All shapes are placed on a 50×50 stud baseplate.
- Dimensions are in LEGO units: studs (X/Z) and plates (Y).
