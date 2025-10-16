"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import bisect
from typing import TypeVar

import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from fairchem.core.datasets.base_dataset import create_dataset

# This draws highly from fairseq's ConcatDataset implementation
# https://github.com/facebookresearch/fairseq/blob/nllb/fairseq/data/concat_dataset.py

T_co = TypeVar("T_co", covariant=True)


class ConcatDataset(Dataset[T_co]):
    @staticmethod
    def cumsum(sequence, sample_ratios):
        r, s = [], 0
        for e, ratio in zip(sequence, sample_ratios):
            curr_len = int(ratio * len(e))
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, datasets, sampling: dict):
        super().__init__()
        if len(datasets) == 0:
            raise RuntimeError(
                "datasets should not be an empty iterable. at least one dataset must have 'key_mappings'"
            )

        # convert the dictionary into two same order lists
        self.dataset_names = list(datasets.keys())
        self.datasets = list(datasets.values())

        dataset_lens = [len(d) for d in self.datasets]
        self.sample_ratios = self._dataset_sampling(
            dataset_lens, self.dataset_names, sampling
        )
        self.cumulative_sizes = self.cumsum(self.datasets, self.sample_ratios)
        self.real_sizes = dataset_lens

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        data_object = self.datasets[dataset_idx][sample_idx]

        # Add additional attributes and cast to appropriate types
        # TODO setting data_object.dataset to dataset.split breaks the collater due to the task map
        #  we should clean that up by using a .dataset and .split keys for all data objects
        data_object.dataset_name = self.dataset_names[dataset_idx]
        # TODO this should be ensured in the datasets themselves, rather than casting/redifining here?
        data_object.fixed = data_object.fixed.long()
        data_object.atomic_numbers = data_object.atomic_numbers.long()

        return data_object

    def _get_dataset_and_sample_index_list(self, sample_idxs: list):
        sample_idxs = np.array(sample_idxs)

        # find out which dataset owns which sample_idx and what the internal idx is for each
        dataset_idx_ownership = np.zeros(sample_idxs.shape[0], dtype=np.int64)
        internal_sample_idxs = np.zeros(sample_idxs.shape[0], dtype=np.int64)
        for dataset_idx in range(len(self.cumulative_sizes)):
            # find out what dataset owns this idx
            this_dataset_idx_ownership = np.where(
                sample_idxs < self.cumulative_sizes[dataset_idx]
            )[0]
            dataset_idx_ownership[this_dataset_idx_ownership] = dataset_idx

            # figure out the offset needed to subtract to get internal index
            offset = 0
            if dataset_idx > 0:
                offset = self.cumulative_sizes[dataset_idx - 1]

            # update internal idx
            internal_sample_idxs[this_dataset_idx_ownership] = (
                sample_idxs[this_dataset_idx_ownership] - offset
            ) % self.real_sizes[dataset_idx]

            # set the just consumed idxs to out of bounds
            sample_idxs[this_dataset_idx_ownership] = self.cumulative_sizes[-1] + 1

        return dataset_idx_ownership, internal_sample_idxs

    # @functools.cache
    # B019 Use of `functools.lru_cache` or `functools.cache` on methods can lead to memory leaks
    def _get_dataset_and_sample_index(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return dataset_idx, sample_idx

    @property
    def updated_dataset_sizes(self):
        new_sizes = [
            (
                self.cumulative_sizes[i]
                if i == 0
                else self.cumulative_sizes[i] - self.cumulative_sizes[i - 1]
            )
            for i, _ in enumerate(self.cumulative_sizes)
        ]
        return (self.real_sizes, new_sizes, self.sample_ratios)

    def metadata_hasattr(self, attr) -> bool:
        for dataset in self.datasets:  # noqa: SIM110
            if not dataset.metadata_hasattr(attr):
                return False
        return True

    def get_metadata(self, attr, sample_idxs_to_get_metadata_for):
        assert attr == "natoms"
        if isinstance(sample_idxs_to_get_metadata_for, list):
            metadata = np.zeros(len(sample_idxs_to_get_metadata_for), dtype=np.int32)
            dataset_idxs, dataset_internal_sample_idx = (
                self._get_dataset_and_sample_index_list(sample_idxs_to_get_metadata_for)
            )
            for dataset_idx in range(len(self.cumulative_sizes)):
                dataset_mask = dataset_idxs == dataset_idx
                metadata[dataset_mask] = self.datasets[dataset_idx].get_metadata(
                    "natoms", list(dataset_internal_sample_idx[dataset_mask])
                )[0]
            return metadata
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(
            sample_idxs_to_get_metadata_for
        )
        return self.datasets[dataset_idx].get_metadata("natoms", sample_idx)

    @staticmethod
    def _dataset_sampling(
        dataset_sizes: list[int], dataset_names: list[str], sampling: dict
    ) -> list[float]:
        """
        Return expansion ratios for each dataset based on sampling strategy
        """
        stype = sampling["type"]

        if stype == "explicit":
            ratios = sampling["ratios"]
            for dataset_name in dataset_names:
                if dataset_name not in ratios:
                    raise ValueError(
                        f"Missing ratio for dataset with name: {dataset_name}"
                    )
            # list comprehension rather than loop for better perf
            return [ratios[dataset_name] for dataset_name in dataset_names]

        elif stype == "balanced":
            indv_target_size = max(dataset_sizes)
            return [indv_target_size / size for size in dataset_sizes]

        elif stype == "temperature":
            temp = sampling["temperature"]
            assert (
                temp >= 1.0
            ), "Temperature must be >= 1.0, for custom weights use weighted sampling."
            # Use numpy for all calculations, reduces python overhead
            dsizes = np.array(dataset_sizes, dtype=np.float64)
            total_size = dsizes.sum()
            temp_prob = np.power(dsizes / total_size, 1.0 / temp)
            temp_prob /= temp_prob.sum()
            max_idx = np.argmax(dsizes)
            target_size = dsizes[max_idx] / temp_prob[max_idx]
            ratios = (target_size * temp_prob) / dsizes
            return ratios.tolist()

        elif stype == "weighted":
            return sampling["ratios"]

        else:
            raise NotImplementedError(f"{sampling} not implemented.")


# This version is used for the Hydra configs
def create_concat_dataset(
    dataset_configs: DictConfig, combined_dataset_config: dict
) -> ConcatDataset:
    """Make a concat dataset with all the splits for each dataset. Keys will be {dataset}.{split}"""
    datasets: dict[str, Dataset] = {}
    # we need to convert omegaConf DictConfig objects to a straight up dict
    dataset_configs = OmegaConf.to_object(dataset_configs)
    for dataset_name in sorted(dataset_configs.keys()):
        dataset_config = dataset_configs[dataset_name]
        assert (
            dataset_config.get("lin_ref", None) is None
        ), "lin_refs in the dataset config are deprecated, please move them to the task config"
        for split in dataset_config["splits"]:
            try:
                datasets[f"{dataset_name}.{split}"] = create_dataset(
                    dataset_config, split=split
                )
            except ValueError as e:
                raise ValueError(f"Error creating dataset '{dataset_name}' {e}") from e

    return ConcatDataset(datasets, sampling=combined_dataset_config["sampling"])
