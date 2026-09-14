# I/O

See [I/O and format support](../io-format-support.md) for the reader
contract, the full format-support table, and exactly what `ISMRMRDReader`
does and does not handle yet.

## Dispatch

::: unimri.io.read

::: unimri.io.Reader

::: unimri.io.register_reader

::: unimri.io.available_readers

::: unimri.io.find_reader

## Readers

`ISMRMRDReader` is the only reader with a real `read()` today (wraps the
`ismrmrd` package; needs `pip install "unimri[ismrmrd]"`). `HDF5Reader` and
`TwixReader` are registered (so `can_read`/format-detection works) but their
`read()` still raises `NotImplementedError` — see the
[roadmap](../roadmap.md).

::: unimri.io.ISMRMRDReader
