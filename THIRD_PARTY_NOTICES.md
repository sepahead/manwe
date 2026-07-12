# Third-party notices

## Hugging Face Candle YOLOv8 example

The YOLOv8 graph and parts of the inference/reporting implementation in
`src/model.rs`, `src/lib.rs`, `src/main.rs`, and `src/coco_classes.rs` are adapted
from Hugging Face Candle's `candle-examples/examples/yolo-v8` example and shared
COCO class table at commit
[`468d1d525fe206a35d6962c02cfa7b9918b31076`](https://github.com/huggingface/candle/tree/468d1d525fe206a35d6962c02cfa7b9918b31076/candle-examples/examples/yolo-v8).
The benchmark crate reuses Manwe's modified implementation.

Candle offers that source under Apache-2.0 or MIT. Manwe uses the MIT option for
the adapted source; see [`LICENSES/Candle-MIT.txt`](LICENSES/Candle-MIT.txt).
The Manwe versions add artifact verification, bounded decoding and reporting,
letterbox inversion, error handling, and tests.

## Roboto Mono

The Rust annotation renderer embeds a stripped subset of **Roboto Mono Regular**
at `src/roboto-mono-stripped.ttf`.

- Original work: Roboto Mono
- Copyright: 2015 The Roboto Mono Project Authors
- Upstream: <https://github.com/googlefonts/robotomono>
- License: Apache License 2.0; see [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)
- Embedded subset SHA-256: `acee6428149e87ccacb2cf66889350c225ec93033dc50c27ec4f315292be860c`

The file is a glyph-reduced derivative retained only for deterministic offline
labels. The project does not claim ownership of the font software or its name.
