"""
variable_datamodule.py
----------------------
Thin LightningDataModule wrapper around mmv_im2im's existing data module
that injects the variable_size_collate_fn into both train and validation
DataLoaders WITHOUT modifying mmv_im2im source code.

--------------
mmv_im2im's DataModule builds its DataLoaders internally, and the YAML
config has no slot for `collate_fn` (it's not a standard serialisable
object).  The cleanest solution is to:
  1. Let the original DataModule set up everything (dataset, transforms,
     split logic, sampler, etc.) as normal.
  2. Override train_dataloader() and val_dataloader() to swap in our
     custom collate function on the DataLoader that would otherwise be
     returned.

This keeps 100 % compatibility with the rest of the training stack.

Usage in run_training.py
------------------------
    from variable_datamodule import VariableSizeDataModule

    base_data_module = get_data_module(cfg.data)   # original mmv_im2im call
    data_module = VariableSizeDataModule(
        base_data_module,
        k=16,             # divisibility target
        mode="constant",  # padding mode
    )
    trainer.fit(model=model, datamodule=data_module)
"""

from typing import Optional

import lightning as pl
from torch.utils.data import DataLoader

from mmv_im2im.utils.variable_collate import make_collate_fn


class VariableSizeDataModule(pl.LightningDataModule):
    """
    Wraps any LightningDataModule and replaces the collate_fn of the
    DataLoaders it produces with variable_size_collate_fn.

    Parameters
    ----------
    base_module : pl.LightningDataModule
        The original data module returned by mmv_im2im.get_data_module().
    k : int
        All spatial dimensions will be padded to multiples of k.
        Use k=16 for AttentionUnet with 4 downsampling stages.
    mode : str
        Padding mode ("constant" recommended – zero-fill after normalisation).
    constant_value : float
        Fill value when mode="constant".
    n_coord_dims : int
        Number of leading elements in the GT vector that hold spatial
        coordinates and must be shifted when padding is applied.
        Default 3 → (z, y, x) centre-of-mass coordinates.
    """

    def __init__(
        self,
        base_module: pl.LightningDataModule,
        k: int = 16,
        mode: str = "constant",
        constant_value: float = 0.0,
        n_coord_dims: int = 3,
    ):
        super().__init__()
        self._base = base_module
        self._collate_fn = make_collate_fn(
            k=k,
            mode=mode,
            constant_value=constant_value,
            n_coord_dims=n_coord_dims,
        )

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def prepare_data(self):
        self._base.prepare_data()

    def setup(self, stage: Optional[str] = None):
        self._base.setup(stage)

    # ------------------------------------------------------------------
    # DataLoader overrides
    # ------------------------------------------------------------------

    def _rebuild_loader(self, original_loader: DataLoader) -> DataLoader:
        """
        Rebuild a DataLoader with the same parameters but our collate_fn.

        We replicate every constructor argument from the existing loader's
        __dict__ so nothing is lost (batch_size, num_workers, pin_memory,
        sampler, etc.).
        """
        init_args = dict(
            dataset=original_loader.dataset,
            batch_size=original_loader.batch_size,
            shuffle=isinstance(
                original_loader.sampler,
                __import__("torch").utils.data.RandomSampler,
            ),
            sampler=(
                original_loader.sampler
                if not isinstance(
                    original_loader.sampler,
                    __import__("torch").utils.data.RandomSampler,
                )
                else None
            ),
            batch_sampler=(
                original_loader.batch_sampler
                if original_loader.batch_sampler is not None
                and not isinstance(
                    original_loader.batch_sampler,
                    __import__("torch").utils.data.BatchSampler,
                )
                else None
            ),
            num_workers=original_loader.num_workers,
            collate_fn=self._collate_fn,  # ← our injection
            pin_memory=original_loader.pin_memory,
            drop_last=original_loader.drop_last,
            timeout=original_loader.timeout,
            worker_init_fn=original_loader.worker_init_fn,
            prefetch_factor=(
                original_loader.prefetch_factor
                if original_loader.num_workers > 0
                else None
            ),
            persistent_workers=original_loader.persistent_workers,
        )

        # Remove None-valued keys that DataLoader doesn't accept as None
        # (batch_sampler=None conflicts with batch_size, for example)
        if init_args["batch_sampler"] is not None:
            # batch_sampler is mutually exclusive with batch_size / shuffle / sampler
            for conflict_key in ("batch_size", "shuffle", "sampler", "drop_last"):
                init_args.pop(conflict_key, None)

        if init_args.get("shuffle") is False and init_args.get("sampler") is None:
            init_args.pop("sampler", None)

        return DataLoader(
            **{
                k: v
                for k, v in init_args.items()
                if v is not None or k in ("batch_size",)
            }
        )

    def train_dataloader(self) -> DataLoader:
        original = self._base.train_dataloader()
        rebuilt = self._rebuild_loader(original)
        print(
            f"[VariableSizeDataModule] train DataLoader rebuilt with "
            f"variable_size_collate_fn  (batch_size={rebuilt.batch_size})"
        )
        return rebuilt

    def val_dataloader(self) -> DataLoader:
        original = self._base.val_dataloader()
        rebuilt = self._rebuild_loader(original)
        print(
            f"[VariableSizeDataModule] val DataLoader rebuilt with "
            f"variable_size_collate_fn  (batch_size={rebuilt.batch_size})"
        )
        return rebuilt

    # ------------------------------------------------------------------
    # Forward any attribute not defined here to the base module
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        # Avoid infinite recursion for attributes set in __init__
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._base, name)
