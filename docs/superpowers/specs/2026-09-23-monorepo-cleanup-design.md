# Monorepo Cleanup Design

## Goal

Prepare the perception repository for transfer into the team monorepo by cleaning up manual tests and assets, synchronizing dependencies, and packaging ONNX models correctly.

## Scope

- Keep the root project name `perception`.
- Move the four manual bag-processing scripts out of the installable Python package into `ap1_perception/old_tests/`.
- Rename those scripts so pytest does not identify them as automated tests.
- Replace package-relative imports in the moved scripts with imports from `ap1_perception`.
- Preserve the scripts as manual diagnostics; do not convert them to automated tests in this change.
- Remove both YOLO `.pt` models because runtime inference will use ONNX exclusively.
- Make `ap1_perception/ap1_perception/yolo/yolo11n.onnx` the default YOLO model.
- Include the YOLO and UFLD ONNX models in installed package data.
- Continue storing `*.onnx` through Git LFS.
- Remove the conflicting `*.onnx` ignore rule.
- Remove demo images, stop-sign frame captures, demo entry points that depend on those captures, and untracked cache residue.
- Regenerate `uv.lock` from the unchanged project name and current `pyproject.toml`.

## File Layout

Manual scripts will live under `ap1_perception/old_tests/`, outside the importable `ap1_perception` package. Their names will describe them as manual bag tools rather than automated tests.

Runtime model assets will remain adjacent to their inference implementations:

- `ap1_perception/ap1_perception/yolo/yolo11n.onnx`
- `ap1_perception/ap1_perception/ufld/model.onnx`

Both paths will be declared in `setup.py` package data and covered by the repository's `*.onnx` Git LFS rule.

## Runtime Behavior

`YOLO()` will default to `yolo11n.onnx` when no model path is supplied. Callers may continue supplying another explicit model path. No runtime path will reference a `.pt` file.

## Removed Material

The cleanup will remove:

- `yolo11n.pt`
- `yolo11n-seg.pt`
- UFLD demo PNG files
- YOLO demo JPEG files
- the `stop_sign_frames/` capture tree
- the YOLO `__main__.py` demo tied to captured frames
- untracked `RANSAC/__pycache__` residue

The UFLD command-line demo will also be removed because its defaults and purpose depend on demo media that is leaving the repository.

## Verification

The implementation is complete when:

1. `uv lock --check` succeeds.
2. No tracked or untracked `.pt`, demo image, stop-sign capture, or stale cache files remain in scope.
3. Both ONNX files remain matched by `.gitattributes`.
4. The default YOLO model path resolves to `yolo11n.onnx`.
5. Built package metadata includes both ONNX assets.
6. Manual scripts import the installed package correctly and are not named as pytest tests.
7. The root project remains named `perception`.
