##
#
# Buffered HDF5 logger for logging experiments.
#
##

# standard imports
import numpy as np
import h5py
from pathlib import Path


#######################################################################
# Logger Class
#######################################################################

class Logger:

    def __init__(self, file_path: str,
                       N: int,
                       dataset_name: str = "data",
                       dtype=np.float32):

        # copy arguments to instance variables
        self.file_path = file_path         # location to save to
        self.N = N                         # vector size
        self.dataset_name = dataset_name   # dataset name inside the HDF5 file

        #  private variables for buffering and state
        self._dtype = np.dtype(dtype) # data type to log
        self._buffer: list = []       # in-memory buffer for rows before dumping to disk
        self._closed = False          # flag to close the logger

        # make sure the parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # number of rows currently buffered in memory (not yet on disk)
    def __len__(self) -> int:
        return len(self._buffer)

    # append one row of length N to the in-memory buffer
    def log(self, data) -> None:
        if self._closed:
            raise RuntimeError("Cannot log to a closed Logger.")

        row = np.asarray(data)
        if row.shape != (self.N,):
            raise ValueError(f"Logger expected row of shape ({self.N},), got {row.shape}.")

        self._buffer.append(row.astype(self._dtype, copy=False))

    # flush the in-memory buffer to the HDF5 file (append, not overwrite)
    def dump(self) -> None:
        if not self._buffer:
            return

        block = np.stack(self._buffer, axis=0)  # shape (n_rows, N)
        self._buffer.clear()

        with h5py.File(self.file_path, "a") as f:
            if self.dataset_name not in f:
                ds = f.create_dataset(
                    self.dataset_name,
                    shape=(0, self.N),
                    maxshape=(None, self.N),
                    dtype=block.dtype,
                    chunks=(min(256, max(16, len(block))), self.N),
                )
            else:
                ds = f[self.dataset_name]

            n = ds.shape[0]
            ds.resize((n + len(block), self.N))
            ds[n:] = block

    # flush any remaining buffered rows and mark the Logger as closed.
    def close(self) -> None:
        if self._closed:
            return
        self.dump()
        self._closed = True
