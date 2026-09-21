from pathlib import Path
import os
import h5py
import numpy as np

class CaudalDataset:
    def __init__(self, path, split="train"):
        self.path = str(Path(path).resolve())
        self._file = None
        self._pid = None
        with h5py.File(self.path, "r") as f:
            if "y" in f:
                if split not in ("train", "validation", "all"):
                    raise ValueError("split debe ser train, validation o all")
                flags = f["split"][:]
                self.ids = np.arange(len(flags)) if split == "all" else np.flatnonzero(flags == (0 if split == "train" else 1))
            else:
                if split not in ("test", "all"):
                    raise ValueError("Para test.h5 usa split='test'")
                self.ids = np.arange(len(f["X"]))

    def _open(self):
        if self._file is None or self._pid != os.getpid():
            self.close()
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._file

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        f = self._open()
        i = int(self.ids[index])
        item = {"Id": i, "X": f["X"][i], "basin_id": int(f["basin_id"][i])}
        if "y" in f:
            item["y"] = f["y"][i]
            item["y_aux"] = f["y_aux"][i]
        return item

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def __del__(self):
        self.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--split", default="train")
    a = p.parse_args()
    data = CaudalDataset(a.path, a.split)
    print("Muestras:", len(data))
    if len(data):
        for k, v in data[0].items():
            print(k, v.shape if hasattr(v, "shape") else v)
    data.close()
