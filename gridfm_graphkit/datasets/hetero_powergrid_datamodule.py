import json
import torch
import os
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
import torch.distributed as dist
from gridfm_graphkit.io.registries import DATASET_WRAPPER_REGISTRY
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    load_normalizer,
    get_task_transforms,
)
from gridfm_graphkit.datasets.utils import (
    split_dataset,
    split_dataset_by_load_scenario_idx,
)
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.datasets.temporal_dataset import HeteroGridTemporalDataset
from gridfm_graphkit.datasets.transforms import (
    RemoveInactiveBranches,
    RemoveInactiveGenerators,
)
import numpy as np
import random
import warnings
import lightning as L
from typing import List, Tuple
from lightning.pytorch.loggers import MLFlowLogger


# Tasks whose dataloading needs the [N, T, F] temporal pipeline rather than
# the static per-scenario pipeline. Centralised here so adding a new
# temporal task only touches one line.
_TEMPORAL_TASK_NAMES = {"TemporalReconstruction"}


class LitGridHeteroDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for power grid datasets.

    This datamodule handles loading, preprocessing, splitting, and batching
    of power grid graph datasets (`GridDatasetDisk`) for training, validation,
    testing, and prediction. It ensures reproducibility through fixed seeds.

    Args:
        args (NestedNamespace): Experiment configuration.
        data_dir (str, optional): Root directory for datasets. Defaults to "./data".

    Attributes:
        batch_size (int): Batch size for all dataloaders. From ``args.training.batch_size``
        data_normalizers (list): List of data normalizers, one per dataset.
        datasets (list): Original datasets for each network.
        train_datasets (list): Train splits for each network.
        val_datasets (list): Validation splits for each network.
        test_datasets (list): Test splits for each network.
        train_dataset_multi (ConcatDataset): Concatenated train datasets for multi-network training.
        val_dataset_multi (ConcatDataset): Concatenated validation datasets for multi-network validation.
        _is_setup_done (bool): Tracks whether `setup` has been executed to avoid repeated processing.

    Methods:
        setup(stage):
            Load and preprocess datasets, split into train/val/test, and store normalizers.
            Handles distributed preprocessing safely.
        train_dataloader():
            Returns a DataLoader for concatenated training datasets.
        val_dataloader():
            Returns a DataLoader for concatenated validation datasets.
        test_dataloader():
            Returns a list of DataLoaders, one per test dataset.
        predict_dataloader():
            Returns a list of DataLoaders, one per test dataset for prediction.

    Notes:
        - Preprocessing is only performed on rank 0 in distributed settings.
        - Subsets and splits are deterministic based on the provided random seed.
        - Normalizers are loaded for each network independently.
        - Test and predict dataloaders are returned as lists, one per dataset.

    Example:
        ```python
        from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule
        from gridfm_graphkit.io.param_handler import NestedNamespace
        import yaml

        with open("config/config.yaml") as f:
            base_config = yaml.safe_load(f)
        args = NestedNamespace(**base_config)

        datamodule = LitGridDataModule(args, data_dir="./data")

        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()
        ```
    """

    def __init__(
        self,
        args: NestedNamespace,
        data_dir: str = "./data",
        normalizer_stats_path: str = None,
        dataset_wrapper: str = None,
        dataset_wrapper_cache_dir: str = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_wrapper = dataset_wrapper
        self.dataset_wrapper_cache_dir = dataset_wrapper_cache_dir
        self.batch_size = int(args.training.batch_size)
        self.split_by_load_scenario_idx = getattr(
            args.data,
            "split_by_load_scenario_idx",
            False,
        )
        self.args = args
        self.normalizer_stats_path = normalizer_stats_path
        self.data_normalizers = []
        self.datasets = []
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.train_scenario_ids: List[List[int]] = []
        self.val_scenario_ids: List[List[int]] = []
        self.test_scenario_ids: List[List[int]] = []
        self._is_setup_done = False

    def setup(self, stage: str):
        if self._is_setup_done:
            print(f"Setup already done for stage={stage}, skipping...")
            return

        # Load pre-fitted normalizer stats if provided (e.g. from a training run)
        saved_stats = None
        if self.normalizer_stats_path is not None:
            saved_stats = torch.load(
                self.normalizer_stats_path,
                map_location="cpu",
                weights_only=True,
            )
            print(f"Loaded normalizer stats from {self.normalizer_stats_path}")

        is_temporal = self._is_temporal_task()

        for i, network in enumerate(self.args.data.networks):
            data_normalizer = load_normalizer(args=self.args)
            self.data_normalizers.append(data_normalizer)

            data_path_network = os.path.join(self.data_dir, network)

            if is_temporal:
                (
                    base_dataset,
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    train_scenario_ids,
                    val_scenario_ids,
                    test_scenario_ids,
                    num_scenarios,
                ) = self._setup_network_temporal(
                    i, network, data_path_network, data_normalizer,
                )
            else:
                (
                    base_dataset,
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    train_scenario_ids,
                    val_scenario_ids,
                    test_scenario_ids,
                    num_scenarios,
                ) = self._setup_network_static(
                    i, network, data_path_network, data_normalizer,
                )

            self.datasets.append(base_dataset)

            # Fit normalizer: restore from saved stats only for fit_on_train
            # normalizers (global baseMVA must match the model's training run).
            # fit_on_dataset normalizers compute per-scenario stats and must
            # always fit on the actual scenarios being used.
            use_saved = (
                saved_stats is not None
                and network in saved_stats
                and data_normalizer.fit_strategy == "fit_on_train"
            )
            if use_saved:
                print(f"Restoring normalizer for {network} from saved stats")
                data_normalizer.fit_from_dict(saved_stats[network])
            else:
                self._fit_normalizer(
                    data_normalizer,
                    data_path_network,
                    network,
                    train_scenario_ids,
                    val_scenario_ids,
                    test_scenario_ids,
                    num_scenarios,
                    saved_stats,
                )

            # Populate the wrapper cache now that the normalizer is fitted,
            # so transform() has BaseMVA set when __getitem__ is called.
            # (Only relevant on the static path; the temporal path doesn't use
            # DATASET_WRAPPER_REGISTRY.)
            if (
                not is_temporal
                and self.dataset_wrapper is not None
                and hasattr(train_dataset, "_setup_cache")
            ):
                train_dataset._setup_cache()

            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            self.test_datasets.append(test_dataset)
            self.train_scenario_ids.append(train_scenario_ids)
            self.val_scenario_ids.append(val_scenario_ids)
            self.test_scenario_ids.append(test_scenario_ids)

        self.train_dataset_multi = ConcatDataset(self.train_datasets)
        self.val_dataset_multi = ConcatDataset(self.val_datasets)
        self._is_setup_done = True

        # Save scenario splits (rank 0 only in DDP)
        is_rank0 = (
            not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        )
        if (
            is_rank0
            and self.trainer is not None
            and getattr(self.trainer, "logger", None) is not None
        ):
            logger = self.trainer.logger
            if isinstance(logger, MLFlowLogger):
                log_dir = os.path.join(
                    logger.save_dir,
                    logger.experiment_id,
                    logger.run_id,
                    "artifacts",
                    "stats",
                )
            else:
                log_dir = os.path.join(logger.save_dir, "stats")
            self.save_scenario_splits(log_dir)

    def _is_temporal_task(self) -> bool:
        """Whether the configured task uses the [N, T, F] temporal pipeline."""
        task_cfg = getattr(self.args, "task", None)
        if task_cfg is None:
            return False
        return getattr(task_cfg, "task_name", "") in _TEMPORAL_TASK_NAMES

    def _setup_network_static(
        self, i: int, network: str, data_path_network: str, data_normalizer,
    ) -> Tuple:
        """Build the per-scenario static pipeline for one network.

        Returns the unwrapped base dataset, the three split subsets, the
        per-split scenario-ID lists, and the resolved ``num_scenarios``.
        """
        is_distributed = dist.is_available() and dist.is_initialized()

        if not is_distributed or dist.get_rank() == 0:
            dataset = HeteroGridDatasetDisk(
                root=data_path_network,
                data_normalizer=data_normalizer,
                transform=get_task_transforms(args=self.args),
            )

        if is_distributed:
            dist.barrier()

        if is_distributed and dist.get_rank() != 0:
            dataset = HeteroGridDatasetDisk(
                root=data_path_network,
                data_normalizer=data_normalizer,
                transform=get_task_transforms(args=self.args),
            )

        base_dataset = dataset

        num_scenarios = self.args.data.scenarios[i]
        if num_scenarios > len(dataset):
            warnings.warn(
                f"Requested number of scenarios ({num_scenarios}) exceeds dataset size ({len(dataset)}). "
                "Using the full dataset instead.",
            )
            num_scenarios = len(dataset)

        all_indices = list(range(len(dataset)))
        # Seed before every shuffle for reproducibility regardless of the
        # order in which power grid datasets are processed.
        random.seed(self.args.seed)
        random.shuffle(all_indices)
        subset_indices = all_indices[:num_scenarios]

        load_scenarios = dataset.load_scenarios[subset_indices]

        dataset = Subset(dataset, subset_indices)

        if self.dataset_wrapper is not None:
            wrapper_cls = DATASET_WRAPPER_REGISTRY.get(self.dataset_wrapper)
            dataset = wrapper_cls(dataset, cache_dir=self.dataset_wrapper_cache_dir)

        np.random.seed(self.args.seed)
        if self.split_by_load_scenario_idx:
            train_dataset, val_dataset, test_dataset = (
                split_dataset_by_load_scenario_idx(
                    dataset,
                    self.data_dir,
                    load_scenarios,
                    self.args.data.val_ratio,
                    self.args.data.test_ratio,
                )
            )
        else:
            train_dataset, val_dataset, test_dataset = split_dataset(
                dataset,
                self.data_dir,
                self.args.data.val_ratio,
                self.args.data.test_ratio,
            )

        train_scenario_ids = self._extract_scenario_ids(
            train_dataset, subset_indices,
        )
        val_scenario_ids = self._extract_scenario_ids(
            val_dataset, subset_indices,
        )
        test_scenario_ids = self._extract_scenario_ids(
            test_dataset, subset_indices,
        )

        return (
            base_dataset,
            train_dataset,
            val_dataset,
            test_dataset,
            train_scenario_ids,
            val_scenario_ids,
            test_scenario_ids,
            num_scenarios,
        )

    def _setup_network_temporal(
        self, i: int, network: str, data_path_network: str, data_normalizer,
    ) -> Tuple:
        """Build the temporal-window pipeline for one network.

        Stacks ``window_size`` consecutive scenarios into ``[N, T, F]``
        samples via :class:`HeteroGridTemporalDataset`, applies the masking
        transforms after stacking, and splits on *windows* (not scenarios).

        The per-scenario transform is restricted to entity cleanup
        (``RemoveInactiveBranches``, ``RemoveInactiveGenerators``) so that
        all scenarios in a window share identical topology and feature
        widths — required by the wrapper's QSTS invariants. The masking
        transforms (``AddTemporalMask`` + ``ApplyMasking``) operate on the
        assembled ``[N, T, F]`` tensors and run inside the temporal
        wrapper's ``transform`` hook.

        Scenarios are taken in the smallest-``load_scenario_idx``-first
        order to satisfy the wrapper's contiguity assertion. Window
        splitting uses the existing random-window split; with overlapping
        windows (stride < window_size) some scenarios appear in multiple
        splits, which is acceptable for masked reconstruction. A proper
        temporal-block split for forecasting evaluation is a separate
        concern handled at evaluation time.
        """
        is_distributed = dist.is_available() and dist.is_initialized()

        # The per-scenario transform strips B_ON / G_ON columns (and the
        # rows of any inactive branches/gens). For QSTS data with fixed
        # topology every entry is active, so this is a column trim
        # uniform across scenarios — the temporal wrapper's topology-
        # invariance check then passes.
        per_scenario_transform = Compose(
            [RemoveInactiveBranches(), RemoveInactiveGenerators()],
        )

        if not is_distributed or dist.get_rank() == 0:
            base_dataset = HeteroGridDatasetDisk(
                root=data_path_network,
                data_normalizer=data_normalizer,
                transform=per_scenario_transform,
            )

        if is_distributed:
            dist.barrier()

        if is_distributed and dist.get_rank() != 0:
            base_dataset = HeteroGridDatasetDisk(
                root=data_path_network,
                data_normalizer=data_normalizer,
                transform=per_scenario_transform,
            )

        num_scenarios = self.args.data.scenarios[i]
        if num_scenarios > len(base_dataset):
            warnings.warn(
                f"Requested number of scenarios ({num_scenarios}) exceeds "
                f"dataset size ({len(base_dataset)}). Using the full "
                "dataset instead.",
            )
            num_scenarios = len(base_dataset)

        # Take the first `num_scenarios` in load_scenario_idx order — this
        # gives a temporally contiguous block, which the wrapper requires.
        full_load = base_dataset.load_scenarios
        order = torch.argsort(full_load)
        contiguous_indices: List[int] = order[:num_scenarios].tolist()
        contiguous_load = full_load[contiguous_indices]

        window_size_attr = getattr(self.args.data, "window_size", None)
        if window_size_attr is None:
            raise ValueError(
                "args.data.window_size is required for the temporal task; "
                "set it in the YAML config (e.g. window_size: 6).",
            )
        window_size = int(window_size_attr)
        window_stride = int(getattr(self.args.data, "window_stride", 1))

        window_transform = get_task_transforms(args=self.args)

        scenario_subset = Subset(base_dataset, contiguous_indices)
        temporal_dataset = HeteroGridTemporalDataset(
            base_dataset=scenario_subset,
            load_scenario_idx=contiguous_load,
            window_size=window_size,
            stride=window_stride,
            transform=window_transform,
        )

        np.random.seed(self.args.seed)
        train_dataset, val_dataset, test_dataset = split_dataset(
            temporal_dataset,
            self.data_dir,
            self.args.data.val_ratio,
            self.args.data.test_ratio,
        )

        train_scenario_ids = self._extract_temporal_scenario_ids(
            train_dataset, contiguous_indices, window_size, window_stride,
        )
        val_scenario_ids = self._extract_temporal_scenario_ids(
            val_dataset, contiguous_indices, window_size, window_stride,
        )
        test_scenario_ids = self._extract_temporal_scenario_ids(
            test_dataset, contiguous_indices, window_size, window_stride,
        )

        return (
            base_dataset,
            train_dataset,
            val_dataset,
            test_dataset,
            train_scenario_ids,
            val_scenario_ids,
            test_scenario_ids,
            num_scenarios,
        )

    @staticmethod
    def _extract_temporal_scenario_ids(
        subset: Subset,
        contiguous_indices: List[int],
        window_size: int,
        window_stride: int,
    ) -> List[int]:
        """Map a window-level Subset back to the unique scenario IDs it covers.

        Each window ``w`` in the temporal dataset spans the contiguous
        scenarios at sorted positions ``[w*stride, w*stride+window_size)``.
        The returned list is the sorted union of original-base-dataset
        scenario IDs that appear in any window of ``subset``. With
        overlapping windows (stride < window_size), neighbouring splits
        will share scenarios — by design, since the underlying time series
        is contiguous.
        """
        window_indices = subset.indices
        if isinstance(window_indices, torch.Tensor):
            window_indices = window_indices.flatten().tolist()
        elif not isinstance(window_indices, list):
            window_indices = list(window_indices)

        scenario_set = set()
        for w in window_indices:
            start = w * window_stride
            for j in range(window_size):
                scenario_set.add(contiguous_indices[start + j])
        return sorted(scenario_set)

    @staticmethod
    def _fit_normalizer(
        data_normalizer,
        data_path_network,
        network,
        train_scenario_ids,
        val_scenario_ids,
        test_scenario_ids,
        num_scenarios,
        saved_stats,
    ):
        """
        Fit normalizer from raw data. In distributed settings, only rank 0
        reads the parquet files and computes stats; the result is broadcast
        to all other ranks via fit_from_dict.
        """
        is_distributed = dist.is_available() and dist.is_initialized()
        is_rank0 = not is_distributed or dist.get_rank() == 0

        raw_data_path = os.path.join(data_path_network, "raw")
        stats = None

        if is_rank0:
            if data_normalizer.fit_strategy == "fit_on_train":
                if saved_stats is not None and network not in saved_stats:
                    warnings.warn(
                        f"No saved normalizer stats found for network '{network}'. "
                        "Fitting from data instead.",
                    )
                print(
                    f"Fitting normalizer on train set ({len(train_scenario_ids)} scenarios)",
                )
                stats = data_normalizer.fit(raw_data_path, train_scenario_ids)
            elif data_normalizer.fit_strategy == "fit_on_dataset":
                all_scenario_ids = (
                    train_scenario_ids + val_scenario_ids + test_scenario_ids
                )
                assert np.unique(all_scenario_ids).shape[0] == num_scenarios
                print(
                    f"Fitting normalizer on full dataset ({len(all_scenario_ids)} scenarios)",
                )
                stats = data_normalizer.fit(raw_data_path, all_scenario_ids)
            else:
                raise ValueError(
                    f"Unknown fit_strategy: {data_normalizer.fit_strategy}",
                )

        if is_distributed:
            stats_list = [stats]
            dist.broadcast_object_list(stats_list, src=0)
            stats = stats_list[0]
            if dist.get_rank() != 0:
                data_normalizer.fit_from_dict(stats)

    @staticmethod
    def _extract_scenario_ids(
        subset: Subset,
        subset_indices: List[int],
    ) -> List[int]:
        """
        Extract original scenario IDs from a Subset.

        The subset's indices point into an outer Subset defined by subset_indices,
        so we map: original_scenario_id = subset_indices[subset_idx].
        """
        indices = subset.indices
        if isinstance(indices, torch.Tensor):
            indices = indices.flatten().tolist()
        elif not isinstance(indices, list):
            indices = list(indices)
        return [subset_indices[idx] for idx in indices]

    def save_scenario_splits(self, log_dir: str):
        """Save train/val/test scenario ID splits to JSON files."""
        os.makedirs(log_dir, exist_ok=True)
        for i, network in enumerate(self.args.data.networks):
            splits = {
                "train": self.train_scenario_ids[i],
                "val": self.val_scenario_ids[i],
                "test": self.test_scenario_ids[i],
            }
            splits_path = os.path.join(log_dir, f"{network}_scenario_splits.json")
            with open(splits_path, "w") as f:
                json.dump(splits, f, indent=2)

    def _dataloader_kwargs(self):
        num_workers = self.args.data.workers
        kwargs = dict(
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        # On Linux some HPC environments restrict passing open file descriptors
        # via Unix socket ancillary data (SCM_RIGHTS), which causes
        # "received 0 items of ancdata" with the default 'fork' start method.
        # 'forkserver' avoids fd-passing by having a dedicated server process
        # that re-opens shared memory objects by name instead.
        if (
            num_workers > 0
            and torch.multiprocessing.get_start_method(allow_none=True) != "spawn"
        ):
            import platform

            if platform.system() == "Linux":
                kwargs["multiprocessing_context"] = "forkserver"
        return kwargs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset_multi,
            batch_size=self.batch_size,
            shuffle=True,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset_multi,
            batch_size=self.batch_size,
            shuffle=False,
            **self._dataloader_kwargs(),
        )

    def test_dataloader(self):
        return [
            DataLoader(
                i,
                batch_size=self.batch_size,
                shuffle=False,
                **self._dataloader_kwargs(),
            )
            for i in self.test_datasets
        ]

    def predict_dataloader(self):
        return [
            DataLoader(
                i,
                batch_size=self.batch_size,
                shuffle=False,
                **self._dataloader_kwargs(),
            )
            for i in self.test_datasets
        ]
