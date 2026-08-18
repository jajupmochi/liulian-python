"""Swiss River dataset adapter built on TS/ST middle interfaces.

Adapted from reference project:
- refer_projects/swiss-river-network-benchmark/swissrivernetwork/benchmark/dataset.py
- refer_projects/swiss-river-network-benchmark/swissrivernetwork/benchmark/train_single_model.py
- refer_projects/swiss-river-network-benchmark/swissrivernetwork/benchmark/test_single_model.py

Provides configurable handling for:
- subsequence breaks (``short_subsequence_method``, ``gap_mode``)
- optional noise injection
- historical target integration (``include_historical_y``)
- nowcasting (``task='nowcast'``)
- LSTM-like full-history mode (``use_full_history=True``)
- entity identifiers (embedding index, one-hot, numeric id, coordinates, sinusoidal)
- graph integration (``graph_mode='edge_index'|'graphlet_features'``)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from liulian.data.backend import ArrayBackend
from liulian.data.scalers import EntityScaler
from liulian.data.spec import TopologySpec
from liulian.data.st.spatialtempodataset import SpatialTempoDataset
from liulian.data.ts.timeseriesdataset import TimeSeriesDataset, TimeSeriesSplit

logger = logging.getLogger(__name__)


class SwissRiverDataset(SpatialTempoDataset):
    """Swiss River dataset with configurable TS and ST adaptation modes.

    Parameters
    ----------
    data_name : str
        Dataset variant: ``'swiss-river-1990'``, ``'swiss-river-2010'``,
        or ``'swiss-river-zurich'``.
    root_path : str or None
        Path to the ``dataset/swiss_river`` directory.  Auto-detected when
        ``None``.
    split_mode : str
        ``'ts'`` — per-station time-series (ConcatDataset-like).
        ``'multi_channel'`` — multi-station multivariate tensor.
    seq_len : int
        Look-back window size (0 for full-sequence mode).
    pred_len : int
        Forecast horizon.
    task : str
        ``'forecast'`` | ``'nowcast'``.
    train_split : float
        Fraction of *training* CSV to use for training (rest → validation).
    use_current_x : bool
        Whether to include the current time-step features as input.
    use_full_history : bool
        When ``True`` each contiguous segment becomes a single sample
        (for LSTM-like models).
    short_subsequence_method : str
        ``'drop'`` | ``'pad'``.
    gap_mode : str
        ``'split'`` | ``'mask_pad'``.
    max_mask_consecutive : int
        Maximum gap to fill when ``gap_mode='mask_pad'``.
    noise_type : str or None
        Noise type (see :mod:`liulian.data.noise`).
    noise_kwargs : dict or None
        Extra noise parameters.
    include_historical_y : str
        ``'none'`` | ``'gt'`` | ``'predicted'``.
    include_historical_predicted_y : bool
        Append predicted y of neighbours as extra features.
    identifier_mode : str
        Entity identifier strategy.
    id_integration : str
        How entity feats are added (``'concat_to_x'`` | ``'add_to_x'`` |
        ``'add_after_patch'``).
    graph_mode : str
        ``'none'`` | ``'edge_index'`` | ``'adj_matrix'`` |
        ``'graphlet_features'``.
    graphlet_num_hops : int
        Neighbourhood radius for graphlet features.
    max_samples : int or None
        Cap the number of samples per split (for debugging / dev runs).
    backend : str or ArrayBackend
        ``'numpy'`` (default) or ``'torch'``.  Controls the format of
        graph arrays (``edge_index``, ``adj_matrix``) and the output
        ``DataSplit`` arrays.
    """

    domain = 'hydrology'
    version = '2.0'

    def __init__(
        self,
        data_name: str = 'swiss-river-1990',
        root_path: Optional[str] = None,
        *,
        split_mode: str = 'per_entity',
        seq_len: int = 90,
        pred_len: int = 1,
        task: str = 'forecast',
        train_split: float = 0.8,
        scaler_type: str = 'none',
        use_current_x: bool = True,
        use_full_history: bool = False,
        short_subsequence_method: str = 'drop',
        gap_mode: str = 'split',
        max_mask_consecutive: int = 10,
        noise_type: str | None = None,
        noise_kwargs: Mapping[str, Any] | None = None,
        include_historical_y: str = 'none',
        include_historical_predicted_y: bool = False,
        identifier_mode: str = 'none',
        id_integration: str = 'concat_to_x',
        # 'model' (canonical) -> transparent identifiers injected by
        # EntityTransparentWrapper (rebuilt per HPO trial); 'data' -> legacy
        # bake-in at window-build time. Both produce bitwise-identical
        # model inputs (see test_transparent_injection_equivalence.py).
        id_injection: str = 'data',
        # sinusoidal_dim / random_identifier_dim: inner-HPO tunable in BOTH
        # split modes (since 2026-06-12). per_entity trials rebuild this
        # dataset + its loaders with the sampled dim (loaders_factory in
        # optim/ray_optimizer.make_trainable); multi_channel rebuilds the
        # ChannelTransparentWrapper instead. The CASE-2 outer sweep (ledger
        # #26) remains for systematic dim-vs-metrics curves.
        sinusoidal_dim: int = 16,
        random_identifier_dim: int = 16,
        random_identifier_seed: int = 2026,
        graph_mode: str = 'none',
        graphlet_num_hops: int = 1,
        max_samples: Optional[int] = None,
        holdout_stations: Optional[Sequence[str]] = None,
        backend: str | ArrayBackend = 'numpy',
    ) -> None:
        self.data_name = data_name
        self.split_mode = split_mode
        # Cold-start / ungauged-station holdout (2026-08-18). Station ids listed
        # here are REMOVED from the train and val windows but KEPT in the test
        # windows; station_ids stays FULL so entity indices, num_entities, and
        # the one-hot/embedding slot count are unchanged (a held-out station's
        # numeric row is simply never trained — that is the point of the
        # comparison). Regime A ("new station has its own short history"): the
        # per-station scaler is still fit on the station's train-range data, so
        # normalization uses the station's own statistics, which a freshly
        # instrumented station would have. Selection/stratification is done at
        # config-authoring time (project rule 3: data lives in config), so this
        # takes an explicit, reproducible list rather than a fraction+seed.
        self.holdout_stations: frozenset[str] = frozenset(str(s) for s in (holdout_stations or []))
        self.train_split = train_split
        self.scaler_type = scaler_type.strip().lower()
        self.graphlet_num_hops = graphlet_num_hops
        self.max_samples = max_samples
        self.sinusoidal_dim = int(sinusoidal_dim)
        self.random_identifier_dim = int(random_identifier_dim)
        self.random_identifier_seed = int(random_identifier_seed)
        self.id_injection = str(id_injection).strip().lower()

        project_root = Path(__file__).resolve().parents[2]
        self.root_path = Path(root_path) if root_path else project_root / 'dataset' / 'swiss_river'

        self._file_map = {
            'swiss-river-1990': {
                'train': 'swiss-1990_train.csv',
                'test': 'swiss-1990_test.csv',
                'graph': 'graph_swiss-1990.pth',
                'graph_name': 'swiss-1990',
            },
            'swiss-river-2010': {
                'train': 'swiss-2010_train.csv',
                'test': 'swiss-2010_test.csv',
                'graph': 'graph_swiss-2010.pth',
                'graph_name': 'swiss-2010',
            },
            'swiss-river-zurich': {
                'train': 'zurich_train.csv',
                'test': 'zurich_test.csv',
                'graph': 'graph_zurich.pth',
                'graph_name': 'zurich',
            },
        }

        if data_name not in self._file_map:
            raise ValueError(f'Unknown data_name: {data_name!r}')

        # --- Load raw DataFrames ----------------------------------------
        train_df = self._read_csv(self._file_map[data_name]['train'])
        test_df = self._read_csv(self._file_map[data_name]['test'])
        train_df, val_df = TimeSeriesDataset.split_train_val(train_df, train_ratio=train_split)

        self.graph_name = self._file_map[data_name]['graph_name']
        # todo: check if we need to use the original read_graph() to get station ids instead:
        self.station_ids = self._infer_station_ids(train_df)

        # Validate holdout ids against the real station list — a typo must fail
        # loudly, never silently hold out nothing.
        if self.holdout_stations:
            unknown = self.holdout_stations - set(self.station_ids)
            if unknown:
                raise ValueError(
                    f'holdout_stations contains ids not in {data_name}: '
                    f'{sorted(unknown)}; available: {self.station_ids}'
                )
            if len(self.holdout_stations) >= len(self.station_ids):
                raise ValueError(
                    f'holdout_stations ({len(self.holdout_stations)}) would leave no '
                    f'training station (of {len(self.station_ids)}).'
                )

        # --- Normalization -----------------------------------------------
        # EntityScaler handles per-entity fit/transform/inverse,
        # matching the reference project's normalize_isolated_station.
        entity_scaler = EntityScaler(
            entity_ids=self.station_ids,
            scaler_type=self.scaler_type,
            feature_suffixes=['_at'],
            target_suffixes=['_wt'],
        )
        entity_scaler.fit(train_df)
        for df in (train_df, val_df, test_df):
            entity_scaler.transform(df)

        # --- Topology: needed for graph-based modes AND for the
        # 'coordinates' identifier (station coords live in the graph file).
        # Before 2026-06-11 it was only loaded for graph modes, so the
        # coordinates identifier silently received an empty mapping.
        # 'coordinates' is the external-wrapper mode; 'coordinates_embedding' is the
        # timellm/gpt4ts internal-identity coords mode — both read station coords from
        # the graph file, so both must load the topology.
        if graph_mode != 'none' or identifier_mode in ('coordinates', 'coordinates_embedding'):
            topology = self._load_topology(self._file_map[data_name]['graph'])
        else:
            topology = None

        self._split_frames = {
            'train': train_df,
            'val': val_df,
            'test': test_df,
        }
        self._split_cache: dict[str, TimeSeriesSplit] = {}

        # --- Initialize parent (SpatialTempoDataset → TimeSeriesDataset) -
        super().__init__(
            splits={'train': pd.DataFrame({'epoch_day': []})},
            time_col='epoch_day',
            feature_cols=['air_temperature'],
            target_cols=['water_temperature'],
            seq_len=seq_len,
            pred_len=pred_len,
            task=task,
            use_current_x=use_current_x,
            include_historical_y=include_historical_y,
            include_historical_predicted_y=include_historical_predicted_y,
            predicted_y_cols=['water_temperature_hat'],
            use_full_history=use_full_history,
            short_subsequence_method=short_subsequence_method,
            gap_mode=gap_mode,
            max_mask_consecutive=max_mask_consecutive,
            noise_type=noise_type,
            noise_kwargs=noise_kwargs,
            station_ids=self.station_ids,
            identifier_mode=identifier_mode,
            id_integration=id_integration,
            # These MUST be forwarded: TimeSeriesDataset.__init__ assigns
            # self.id_injection / sinusoidal_dim / random_identifier_dim /
            # random_identifier_seed and would silently overwrite our earlier
            # assignments with the parent defaults (same trap as the entity
            # scaler below; bit us 2026-06-12 — custom dims were reset to 16).
            id_injection=id_injection,
            sinusoidal_dim=sinusoidal_dim,
            random_identifier_dim=random_identifier_dim,
            random_identifier_seed=random_identifier_seed,
            topology=topology,
            graph_mode=graph_mode,
            graph_metadata={'graph_name': self.graph_name},
            backend=backend,
        )

        # Restore the entity scaler after super().__init__() which sets
        # self._scaler = None for the generic TimeSeriesDataset path.  Do not move it before super().__init__().
        self._scaler = entity_scaler

    # ------------------------------------------------------------------
    # Scaler delegation
    # ------------------------------------------------------------------

    @property
    def station_scaler(self) -> EntityScaler:
        """The fitted per-station scaler (read-only)."""
        return self._scaler

    @property
    def target_scalers(self) -> Dict[str, Any]:
        """Per-station target scalers: ``station_id → scaler``."""
        return self._scaler.target_scalers

    @property
    def feature_scalers(self) -> Dict[str, Any]:
        """Per-station feature scalers: ``station_id → scaler``."""
        return self._scaler.feature_scalers

    def inverse_transform(self, data, **kwargs):
        """Inverse-transform normalized target values back to original scale.

        Delegates to :class:`EntityScaler.inverse_transform`.
        When ``scaler_type='none'`` this is an identity operation.

        Parameters
        ----------
        data : numpy.ndarray or torch.Tensor
            Shape ``(B, T, C)`` or ``(N, C)``.
        **kwargs
            Forwarded to :meth:`EntityScaler.inverse_transform`.
            Typically ``entity_ids`` (list of station IDs) and/or
            ``timestamps``.

        Returns
        -------
        Same type and shape as *data*, in the original (un-normalized) scale.
        """
        return self._scaler.inverse_transform(data, **kwargs)

    def inverse_transform_station(
        self,
        data: np.ndarray,
        station_id: str,
        suffix: str = '_wt',
    ) -> np.ndarray:
        """Per-station inverse transform (delegates to EntityScaler)."""
        return self._scaler.inverse_transform_entity(data, station_id, suffix)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _read_csv(self, filename: str) -> pd.DataFrame:
        path = self.root_path / filename
        if not path.exists():
            raise FileNotFoundError(f'Dataset CSV not found: {path}')
        return pd.read_csv(path)

    @staticmethod
    def _infer_station_ids(df: pd.DataFrame) -> list[str]:
        """Extract station identifiers from ``*_wt`` column names.

        Note:
            this assumes that training, validation, and test .csv files have the same station column order. This
            order will be used to auto-build entity scalers for the multichannel mode.
        """
        station_cols = [col for col in df.columns if col.endswith('_wt')]
        return [col[:-3] for col in station_cols]

    def _load_topology(self, graph_filename: str) -> TopologySpec:
        """Load graph topology from a ``.pth`` file.

        The ``.pth`` file contains ``(x, edge_index)`` where ``x`` is a
        node feature tensor with station-id in column 2 and optional
        coordinates in columns 0–1, and ``edge_index`` is ``(2, E)``.
        """
        graph_path = self.root_path / graph_filename
        if not graph_path.exists():
            return TopologySpec(node_ids=self.station_ids, edges=[])

        try:
            import torch
        except ImportError:
            return TopologySpec(node_ids=self.station_ids, edges=[])

        x, edge_index = torch.load(graph_path, weights_only=False)

        node_ids = [str(int(i)) for i in x[:, 2].tolist()]
        edges: list[tuple[str, str]] = []
        if hasattr(edge_index, 'shape') and edge_index.shape[0] == 2:
            src = edge_index[0].tolist()
            dst = edge_index[1].tolist()
            for s, d in zip(src, dst):
                if 0 <= s < len(node_ids) and 0 <= d < len(node_ids):
                    edges.append((node_ids[s], node_ids[d]))
                else:
                    raise ValueError(f'Edge index out of bounds: {(s, d)} with {len(node_ids)} nodes.')

        coords: dict[str, tuple[float, float]] = {}
        if hasattr(x, 'shape') and x.shape[1] >= 2:
            for idx, station in enumerate(node_ids):
                coords[station] = (float(x[idx, 0]), float(x[idx, 1]))

        return TopologySpec(node_ids=node_ids, edges=edges, coordinates=coords)

    # ------------------------------------------------------------------
    # Per-station DataFrame construction
    # ------------------------------------------------------------------

    def _make_station_frame(
        self,
        df: pd.DataFrame,
        station: str,
        graphlet_neighbors: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build a single-station DataFrame with standard column names.  # todo: resolve the duplication

        Parameters
        ----------
        df:
            Raw multi-station DataFrame.
        station:
            Station identifier.
        graphlet_neighbors:
            Neighbour station ids whose predicted y should be included as
            extra features (graphlet mode).
        """
        out = pd.DataFrame(
            {
                'epoch_day': df['epoch_day'],
                'air_temperature': df[f'{station}_at'],
                'water_temperature': df[f'{station}_wt'],
            }
        )
        if 'has_nan' in df.columns:
            out['has_nan'] = df['has_nan']

        # Graphlet: add neighbour predictions as extra input features
        if graphlet_neighbors:
            for neigh in graphlet_neighbors:
                hat_col = f'{neigh}_wt_hat'
                raw_col = f'{neigh}_wt'
                if hat_col in df.columns:
                    out[hat_col] = df[hat_col]
                elif raw_col in df.columns:
                    out[hat_col] = df[raw_col]

        # Include predicted y for this station if available
        if f'{station}_wt_hat' in df.columns:
            out['water_temperature_hat'] = df[f'{station}_wt_hat']

        # Drop rows where THIS station has missing measurements (stations
        # join/leave the network at different times — the swiss-river-2010 /
        # zurich CSVs carry NaNs; swiss-river-1990 is pre-cleaned so this is
        # a no-op there). The resulting epoch_day jumps become segment breaks
        # via _detect_breaks(), so no window ever spans a dropped region.
        # Without this a single NaN sample drives the whole per_entity loss
        # to NaN (observed on swiss-river-2010, 2026-06-11).
        out = out.dropna().reset_index(drop=True)

        return out

    # ------------------------------------------------------------------
    # Graphlet neighbour map
    # ------------------------------------------------------------------

    def _graphlet_neighbor_map(self) -> dict[str, list[str]]:
        """Return mapping station → list of k-hop neighbour station ids."""
        if self.topology is None or not self.topology.edges:
            return {station: [] for station in self.station_ids}

        adj: dict[str, list[str]] = {station: [] for station in self.station_ids}
        for src, dst in self.topology.edges:
            if src in adj:
                adj[src].append(dst)
            if dst in adj:
                adj[dst].append(src)

        if self.graphlet_num_hops <= 1:
            return {k: sorted(set(v)) for k, v in adj.items()}

        out: dict[str, list[str]] = {}
        for station in self.station_ids:
            visited = {station}
            frontier = {station}
            for _ in range(self.graphlet_num_hops):
                nxt = set()
                for node in frontier:
                    nxt.update(adj.get(node, []))
                nxt -= visited
                visited |= nxt
                frontier = nxt
            visited.discard(station)
            out[station] = sorted(visited)
        return out

    # ------------------------------------------------------------------
    # Split builders
    # ------------------------------------------------------------------

    def _build_pe_split(self, split_name: str) -> TimeSeriesSplit:
        """Build a per-station time-series split (per_entity mode).

        Each station gets its own :class:`TimeSeriesDataset`, and the
        resulting samples are concatenated (equivalent to
        ``torch.utils.data.ConcatDataset`` in the reference project).
        """
        df = self._split_frames[split_name]
        graphlet_map = self._graphlet_neighbor_map() if self.graph_mode == 'graphlet_features' else {}

        # Cold-start holdout: held-out stations contribute NO train/val windows
        # but ARE evaluated at test. station_ids stays full so entity indices and
        # num_entities are unchanged; only which stations produce windows differs.
        if self.holdout_stations and split_name in ('train', 'val'):
            stations = [s for s in self.station_ids if s not in self.holdout_stations]
            logger.info(
                '[coldstart] %s split: %d/%d stations (held out %d: %s)',
                split_name,
                len(stations),
                len(self.station_ids),
                len(self.holdout_stations),
                sorted(self.holdout_stations),
            )
        else:
            stations = list(self.station_ids)

        parts: list[TimeSeriesSplit] = []
        for station in stations:
            station_df = self._make_station_frame(df, station, graphlet_neighbors=graphlet_map.get(station))
            predicted_cols = [col for col in station_df.columns if col.endswith('_wt_hat')]
            ds_station = TimeSeriesDataset(
                splits={split_name: station_df},
                time_col='epoch_day',
                feature_cols=['air_temperature'] + predicted_cols,
                target_cols=['water_temperature'],
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                task=self.task,
                use_current_x=self.use_current_x,
                include_historical_y=self.include_historical_y,
                include_historical_predicted_y=self.include_historical_predicted_y,
                predicted_y_cols=predicted_cols + ['water_temperature_hat'],
                use_full_history=self.use_full_history,
                short_subsequence_method=self.short_subsequence_method,
                gap_mode=self.gap_mode,
                max_mask_consecutive=self.max_mask_consecutive,
                noise_type=self.noise_type,
                noise_kwargs=self.noise_kwargs,
                station_ids=self.station_ids,
                identifier_mode=self.identifier_mode,
                id_integration=self.id_integration,
                id_injection=self.id_injection,
                coordinates=(self.topology.coordinates if self.topology else {}),
                station_name=station,
                sinusoidal_dim=self.sinusoidal_dim,
                random_identifier_dim=self.random_identifier_dim,
                random_identifier_seed=self.random_identifier_seed,
            )
            parts.append(ds_station.get_split(split_name))

        merged = TimeSeriesSplit.merge(parts, name=split_name)
        return merged.with_max_samples(self.max_samples)

    def _build_mc_split(self, split_name: str) -> TimeSeriesSplit:
        """Build a multi-station multivariate split (multi_channel mode).

        All stations are treated as separate feature / target channels in
        a single :class:`TimeSeriesDataset`, producing samples of shape
        ``(seq_len, N_stations * n_features)`` — the pattern used by
        the [``Time-Series-Library``](https://github.com/thuml/Time-Series-Library/tree/main).
        """
        df = self._split_frames[split_name].copy().reset_index(drop=True)
        feature_cols = [f'{station}_at' for station in self.station_ids if f'{station}_at' in df.columns]
        target_cols = [f'{station}_wt' for station in self.station_ids if f'{station}_wt' in df.columns]

        mc_ds = TimeSeriesDataset(
            splits={split_name: df[['epoch_day'] + feature_cols + target_cols]},
            time_col='epoch_day',
            feature_cols=feature_cols,
            target_cols=target_cols,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            task=self.task,
            use_current_x=self.use_current_x,
            include_historical_y='none',
            include_historical_predicted_y=False,
            use_full_history=self.use_full_history,
            short_subsequence_method=self.short_subsequence_method,
            gap_mode=self.gap_mode,
            max_mask_consecutive=self.max_mask_consecutive,
            noise_type=self.noise_type,
            noise_kwargs=self.noise_kwargs,
            # Entity identifier params — required so that downstream
            # pipeline / adapter code (e.g. EntityAwareMixin,
            # ChannelEntityWrapper, ChannelTransparentWrapper) can
            # detect the configured mode and wrap the model accordingly.
            station_ids=self.station_ids,
            identifier_mode=self.identifier_mode,
            id_integration=self.id_integration,
            coordinates=(self.topology.coordinates if self.topology else {}),
            sinusoidal_dim=self.sinusoidal_dim,
            random_identifier_dim=self.random_identifier_dim,
            random_identifier_seed=self.random_identifier_seed,
        )
        split = mc_ds.get_split(split_name)
        return split.with_max_samples(self.max_samples)

    def _build_ci_split(self, split_name: str) -> TimeSeriesSplit:
        """Build a channel-independent split.

        Uses the TS mode (per-station) base split, then wraps it with
        :class:`~liulian.data.ts.channel_independent.ChannelIndependentDataset`
        so each feature channel is treated as a separate univariate series.
        """
        from liulian.data.ts.channel_independent import ChannelIndependentDataset

        base_split = self._build_pe_split(split_name)
        ci_ds = ChannelIndependentDataset(base_split)
        return ci_ds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_split(self, split_name: str) -> TimeSeriesSplit:
        if split_name not in ('train', 'val', 'test'):
            raise KeyError(f'Unknown split: {split_name!r}. Use train/val/test.')
        if split_name not in self._split_cache:
            if self.split_mode == 'multi_channel':
                self._split_cache[split_name] = self._build_mc_split(split_name)
            elif self.split_mode == 'channel_independent':
                self._split_cache[split_name] = self._build_ci_split(split_name)
            else:
                self._split_cache[split_name] = self._build_pe_split(split_name)
        return self._split_cache[split_name]

    def info(self) -> Dict[str, Any]:
        out = super().info()
        # Derive target names based on split mode
        if self.split_mode == 'multi_channel':
            target_names = [f'{station}_wt' for station in self.station_ids]
        else:
            target_names = ['water_temperature']
        out.update(
            {
                'domain': self.domain,
                'version': self.version,
                'data_name': self.data_name,
                'root_path': str(self.root_path),
                'split_mode': self.split_mode,
                'graph_name': self.graph_name,
                'num_stations': len(self.station_ids),
                'target_names': target_names,
            }
        )
        return out

    # get_data_loaders is inherited from TimeSeriesDataset (unified
    # implementation that handles both 4-tuple and 6-tuple entity-aware
    # batch output).  The collate function auto-detects whether the
    # split includes seg_entity_ids and returns entity_id_strs +
    # entity_idx accordingly.
