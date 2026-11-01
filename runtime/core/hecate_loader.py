"""Virtual tensors for the Hecate model folder (import before the model loads).

The release folder carries no EXL3 trellis for linears that never run from it (int8-served attention / shared experts /
dense MLPs / LM head, and the int4-served routed experts). ExLlamaV3 still builds those modules from the
EXL3 layout, so `virtual.json` lists their names, dtypes and shapes; this module registers them in the tensor collection
as zero-byte entries (the loader's memory budget counts 0) and serves each as a zero-stride placeholder of the right
shape (no memory). Anything that would actually compute with a placeholder is caught by the int8 / int4 paths first; a
placeholder that reaches a trellis kernel is a configuration error and is reported at load (see hecate_runtime)."""
import json, os
import torch

VIRTUAL = os.environ.get("HECATE_VIRTUAL", "")
_TAG = "<virtual>"
_DT = {"I16": torch.int16, "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "I32": torch.int32,
       "U32": torch.uint32 if hasattr(torch, "uint32") else torch.int32}
stats = {"registered": 0, "served": 0}


def _install():
    from exllamav3 import Config
    from exllamav3.loader.safetensors import SafetensorsCollection
    fd = Config.from_directory

    def from_directory(*a, **k):
        c = fd(*a, **k)
        path = VIRTUAL or os.path.join(c.directory, "virtual.json")
        if os.path.exists(path):
            V = json.load(open(path))
            hdr = {}
            for key, v in V.items():
                if key in c.stc.tensor_file_map:        # a real tensor (e.g. the warm band) wins
                    continue
                hdr[key] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [0, 0]}
                c.stc.tensor_file_map[key] = _TAG
            c.stc.file_headers[_TAG] = hdr
            stats["registered"] = len(hdr)
            print(f" -- hecate: {len(hdr)} virtual tensors from {path} (never loaded)", flush=True)
        return c
    Config.from_directory = staticmethod(from_directory)

    gt = SafetensorsCollection.get_tensor

    def get_tensor(self, key, device=None, *a, **k):
        if self.tensor_file_map.get(key) == _TAG:
            v = self.file_headers[_TAG][key]
            stats["served"] += 1
            dev = device if device is not None else "cpu"
            return torch.zeros(1, dtype=_DT[v["dtype"]], device=dev).expand(*v["shape"]) if v["shape"] else \
                torch.zeros((), dtype=_DT[v["dtype"]], device=dev)
        return gt(self, key, device, *a, **k)
    SafetensorsCollection.get_tensor = get_tensor

    # EXL3 codebook markers (.mul1 / .mcg) are read straight from their file once per value; a virtual marker holds the
    # value the loader asks for
    gcm = SafetensorsCollection.get_codebook_marker
    _markers = {}

    def get_codebook_marker(self, key, value):
        if self.tensor_file_map.get(key) == _TAG:
            m = _markers.get(value)
            if m is None:
                m = _markers[value] = torch.tensor(value, dtype=torch.uint32).view(torch.int)
            return m
        return gcm(self, key, value)
    SafetensorsCollection.get_codebook_marker = get_codebook_marker

    # the collection opens a handle per file it serves; the virtual tag has none
    if hasattr(SafetensorsCollection, "get_handle"):
        gh = SafetensorsCollection.get_handle

        def get_handle(self, filename, *a, **k):
            if filename == _TAG:
                raise RuntimeError("hecate: a virtual tensor was read through a file handle (unsupported load path)")
            return gh(self, filename, *a, **k)
        SafetensorsCollection.get_handle = get_handle


_install()
