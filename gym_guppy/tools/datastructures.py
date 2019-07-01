import numpy as np

class LazyFrames:
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.array(self._frames)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]
