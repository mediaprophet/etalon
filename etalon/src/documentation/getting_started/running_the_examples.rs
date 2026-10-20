/*!
# Running the Examples

This section shows you how to get the dependencies needed to run the
examples included in the `etalon` repository.

## Dependencies

The examples require:

- The Rust compiler ([installation instructions](https://www.rust-lang.org/)).
- Drivers for your graphics card. Note that some platforms have multiple
  possible driver backends (e.g. D3D12 or Vulkan on Windows).
- Git ([installation instructions](https://git-scm.com/downloads)).

## Cloning

Once you have the tools above, clone the repository:

```bash
git clone --depth 1 https://github.com/mediaprophet/etalon.git
cd etalon
```

## Running the examples

You can run the examples using the `etalon-examples` binary:

```bash
# Show a list of all examples
cargo run --bin etalon-examples

# Run the cube example
cargo run --bin etalon-examples cube
```

To run an example in a browser, see
[Running on the Web](crate::documentation::platforms::web).
*/
