"""Experiment — top-level orchestrator for the liulian pipeline.

The :class:`Experiment` class wires together a task, dataset, model,
and (optionally) an optimiser and logger, then drives the lifecycle
through the state machine: INIT → TRAIN → EVAL → (INFER) → COMPLETED.

For PyTorch forecasting models it delegates to
:class:`~liulian.runtime.trainer.ForecastTrainer` which encapsulates the
full training loop (gradient optimisation, LR scheduling, early stopping,
checkpoint management).  Experiment scripts stay minimal::

    exp = Experiment(spec, task, dataset, model, data_loaders=loaders)
    summary = exp.run()  # train + eval + final test
    summary = exp.run(train=False)  # evaluate from checkpoint
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional


from liulian.data.base import BaseDataset
from liulian.loggers.interface import LoggerInterface
from liulian.models.base import ExecutableModel
from liulian.optim.base import BaseOptimizer
from liulian.runtime.spec import ExperimentSpec
from liulian.runtime.state_machine import LifecycleState, StateMachine
from liulian.tasks.base import BaseTask
from liulian.utils.helpers import ensure_dir, timestamp_id


logger = logging.getLogger(__name__)


class Experiment:
    """Orchestrates the full experiment lifecycle.

    For lightweight / non-PyTorch models the class runs a single forward
    pass per phase.  When *data_loaders* are supplied (PyTorch DataLoaders
    for train / val / test) the class spawns a
    :class:`~liulian.runtime.trainer.ForecastTrainer` that manages the full
    multi-epoch training loop.

    Usage::

        # Minimal (non-torch) -------------------------------------------------
        exp = Experiment(spec, task, dataset, model)
        summary = exp.run()

        # PyTorch with data loaders -------------------------------------------
        loaders = {'train': train_dl, 'val': val_dl, 'test': test_dl}
        exp = Experiment(spec, task, dataset, model, data_loaders=loaders)
        summary = exp.run()
    """

    def __init__(
        self,
        spec: ExperimentSpec,
        task: BaseTask,
        dataset: BaseDataset,
        model: ExecutableModel,
        optimizer: Optional[BaseOptimizer] = None,
        exp_logger: Optional[LoggerInterface] = None,
        config: Optional[Dict[str, Any]] = None,
        data_loaders: Optional[Dict[str, Any]] = None,
        torch_model: Optional[Any] = None,
    ) -> None:
        """Set up the experiment from its constituent components.

        Args:
            spec: Full experiment specification for reproducibility.
            task: Task instance defining loss and metrics.
            dataset: Dataset providing train/val/test splits.
            model: Model (ExecutableModel adapter) for eval/inference.
            optimizer: Optional HPO engine.
            exp_logger: Optional experiment logger (WandB / local).
            config: Experiment configuration dictionary (training hypers,
                etc.).  Forwarded to the :class:`ForecastTrainer`.
            data_loaders: Dict with ``"train"``, ``"val"``, and optionally
                ``"test"`` PyTorch :class:`~torch.utils.data.DataLoader`
                instances.  When provided, :meth:`run` uses the
                :class:`ForecastTrainer` for multi-epoch training.
            torch_model: Raw ``nn.Module`` for training.  If *None* and
                *model* is a :class:`TorchModelAdapter`, ``model._model``
                is used.
        """
        self.spec = spec
        self.task = task
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.exp_logger = exp_logger
        self.config = config or {}
        self.data_loaders = data_loaders
        self.torch_model = torch_model

        self._sm = StateMachine()
        self._callbacks: Dict[str, List[Callable[..., Any]]] = {
            'on_epoch_end': [],
            'on_eval_end': [],
            'on_checkpoint': [],
            'on_infer_complete': [],
        }
        self._artifacts_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def register_callback(self, event: str, fn: Callable[..., Any]) -> None:
        """Register a callback for *event*.

        Args:
            event: One of ``"on_epoch_end"``, ``"on_eval_end"``,
                ``"on_checkpoint"``, ``"on_infer_complete"``.
            fn: Callable invoked when the event fires.

        Raises:
            KeyError: If *event* is not recognised.
        """
        if event not in self._callbacks:
            raise KeyError(f"Unknown event '{event}'. Valid: {list(self._callbacks)}")
        self._callbacks[event].append(fn)

    def _fire(self, event: str, **kwargs: Any) -> None:
        """Invoke all callbacks registered for *event*."""
        for fn in self._callbacks.get(event, []):
            fn(**kwargs)

    # ------------------------------------------------------------------
    # Inverse-transform resolution (for denormalized eval metrics)
    # ------------------------------------------------------------------

    def _resolve_inverse_transform_fn(
        self,
        loaders: Dict[str, Any],
    ) -> Optional[Callable[[Any], Any]]:
        """Resolve a callable inverse-transform function for eval denorm.

        Resolution order:

        1. ``config['inverse_transform']`` when callable
        2. ``dataset.inverse_transform`` when available
        3. Any loader dataset's ``inverse_transform`` (train/val/test)

        Returns:
            Callable inverse transform or ``None`` if unavailable.
        """
        configured = self.config.get('inverse_transform')
        if callable(configured):
            return configured

        dataset_inv = getattr(self.dataset, 'inverse_transform', None)
        if callable(dataset_inv):
            return dataset_inv

        for split_name in ('train', 'val', 'test'):
            loader = loaders.get(split_name)
            if loader is None:
                continue
            ds = getattr(loader, 'dataset', None)
            split_inv = getattr(ds, 'inverse_transform', None)
            if callable(split_inv):
                return split_inv

        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(
        self,
        train: bool = True,
        eval: bool = True,
        infer: bool = False,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Execute the experiment pipeline.

        If *data_loaders* were provided at construction time, the method
        delegates training and evaluation to :class:`ForecastTrainer`.
        Otherwise it falls back to a single-batch forward-pass approach
        (useful for simple / non-PyTorch models).

        Args:
            train: Whether to execute the training phase.
            eval: Whether to execute evaluation after training.
            infer: Whether to execute inference after evaluation.
            batch_size: Batch size for single-batch fallback mode.

        Returns:
            Summary dictionary with ``"status"``, ``"metrics"``, etc.
        """
        # ---- Artifacts directory ----
        run_id = f'{self.spec.name}_{timestamp_id()}'
        self._artifacts_dir = ensure_dir(os.path.join('artifacts', run_id))

        spec_path = os.path.join(self._artifacts_dir, 'spec.yaml')
        self.spec.to_yaml(spec_path)
        logger.info('Experiment spec saved to %s', spec_path)

        # Auto-create a logger from config if none provided
        if self.exp_logger is None:
            self.exp_logger = self._maybe_create_logger()

        summary: Dict[str, Any] = {
            'status': 'ok',
            'metrics': {},
            'run_id': run_id,
            'artifacts_dir': self._artifacts_dir,
        }

        # ---- Dispatch to torch trainer or simple mode ----
        if self.data_loaders is not None:
            self._run_torch(summary, train=train, eval=eval, infer=infer)
        else:
            self._run_simple(summary, train=train, eval=eval, infer=infer, batch_size=batch_size)

        # ---- Finalise ----
        self._sm.transition(LifecycleState.COMPLETED)
        summary['state'] = self._sm.state.value

        if self.exp_logger:
            self.exp_logger.log_artifact(spec_path)

        # ---- Auto-viz ----
        if self.config.get('auto_viz', False) and 'predictions' in summary:
            try:
                viz_paths = self.visualize(summary)
                summary['viz_paths'] = viz_paths
            except Exception as exc:
                logger.warning('Auto-viz failed: %s', exc)

        logger.ok("Experiment '%s' completed.", self.spec.name)
        return summary

    # ------------------------------------------------------------------
    # Individual phase helpers
    # ------------------------------------------------------------------

    def train(self, **kwargs: Any) -> Dict[str, Any]:
        """Run only the training phase.

        Convenience wrapper for ``run(train=True, eval=False)``.
        """
        return self.run(train=True, eval=False, **kwargs)

    def val(self, **kwargs: Any) -> Dict[str, Any]:
        """Run only the evaluation phase (no training).

        Convenience wrapper for ``run(train=False, eval=True)``.
        """
        return self.run(train=False, eval=True, **kwargs)

    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        """Alias for :meth:`val`."""
        return self.val(**kwargs)

    def visualize(
        self,
        summary: Optional[Dict[str, Any]] = None,
        *,
        output_dir: str | None = None,
        method: str = 'mean',
    ) -> Dict[str, str]:
        """Generate prediction visualisations from experiment results.

        Can be called after :meth:`run` with the returned summary, or
        standalone with pre-loaded predictions.

        Args:
            summary: Experiment summary dict containing ``'predictions'``
                with ``preds``, ``trues``, ``times`` tensors.
            output_dir: Output directory for plots.  Defaults to
                ``{artifacts_dir}/figures``.
            method: Aggregation method for overlapping predictions.

        Returns:
            Mapping of plot name → file path.
        """
        from liulian.viz.plots import save_prediction_plots

        if summary is None:
            raise ValueError('No summary provided — call run() first.')

        predictions = summary.get('predictions', {})
        preds = predictions.get('preds')
        trues = predictions.get('trues')
        times = predictions.get('times')

        if preds is None or trues is None or times is None:
            logger.warning('No predictions in summary — skipping viz.')
            return {}

        # Convert to numpy if tensors
        import torch as _torch

        if isinstance(preds, _torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(trues, _torch.Tensor):
            trues = trues.detach().cpu().numpy()
        if isinstance(times, _torch.Tensor):
            times = times.detach().cpu().numpy()

        if output_dir is None:
            output_dir = os.path.join(self._artifacts_dir or 'artifacts', 'figures')

        viz_method = self.config.get('viz_method', method)
        pred_len = self.config.get('pred_len')

        paths = save_prediction_plots(
            preds,
            trues,
            times,
            entity_ids=predictions.get('entity_ids'),
            method=viz_method,
            pred_len=pred_len,
            output_dir=output_dir,
            title_prefix=f'{self.spec.name} — ',
        )

        logger.ok('Saved %d plots to %s', len(paths), output_dir)
        return paths

    # ------------------------------------------------------------------
    # PyTorch path (ForecastTrainer)
    # ------------------------------------------------------------------

    def _run_torch(
        self,
        summary: Dict[str, Any],
        *,
        train: bool,
        eval: bool,
        infer: bool,
    ) -> None:
        """Drive the lifecycle using :class:`ForecastTrainer`.

        When ``self.optimizer`` is set and a ``search_space`` is present
        in ``self.config``, the method branches to an HPO flow that
        delegates to :meth:`RayOptimizer.run`. Otherwise, it runs a
        single-trial training loop as before.
        """
        from liulian.runtime.trainer import ForecastTrainer

        loaders = self.data_loaders or {}
        train_loader = loaders.get('train')
        val_loader = loaders.get('val')
        test_loader = loaders.get('test')

        # Resolve the raw nn.Module for training
        torch_model = self.torch_model
        if torch_model is None:
            # Try to extract from adapter
            torch_model = getattr(self.model, '_model', None)
        if torch_model is None:
            raise RuntimeError(
                'Cannot find a torch nn.Module.  Pass torch_model= or use a TorchModelAdapter-based model.'
            )

        ckpt_dir = os.path.join(self._artifacts_dir, 'checkpoints')

        # ----- HPO branch -----
        search_space = self.config.get('search_space')
        if train and self.optimizer is not None and search_space and train_loader is not None:
            self._sm.transition(LifecycleState.TRAIN)
            logger.info('Starting HPO via %s', type(self.optimizer).__name__)

            # Detect EntityWrapper / ChannelEntityWrapper and extract the
            # inner model class so that make_trainable can build fresh
            # models per trial.
            from liulian.models.torch.entity_mixin import (
                ChannelEntityWrapper,
                ChannelTransparentWrapper,
                EntityTransparentWrapper,
                EntityWrapper,
            )

            is_entity_wrapped = isinstance(torch_model, EntityWrapper)
            is_channel_wrapped = isinstance(torch_model, ChannelEntityWrapper)
            is_transparent_wrapped = isinstance(torch_model, ChannelTransparentWrapper)
            is_pe_transparent = isinstance(torch_model, EntityTransparentWrapper)

            if is_pe_transparent:
                # Per_entity transparent modes with id_injection='model'. The
                # wrapper holds the (N, D) fixed feature table; capture its
                # rebuild ingredients so a trial-tuned sinusoidal_dim /
                # random_identifier_dim takes effect (the inner model is
                # rebuilt with enc_in = base + tuned_dim).
                inner_model_cls = type(torch_model.inner)
                model_args = getattr(torch_model.inner, '_args', None)
                _pt_mode = str(torch_model.mode)
                _pt_station_ids = list(torch_model.station_ids)
                _pt_coords = torch_model._coordinates
                _pt_seed = int(torch_model.random_seed)
                _pt_fixed_dim = int(torch_model.feature_dim)
            elif is_channel_wrapped:
                inner_model_cls = type(torch_model.inner)
                model_args = getattr(torch_model.inner, '_args', None)
                _cw_num_stations = torch_model.station_embedding.num_embeddings
                _cw_emb_size = torch_model.station_embedding.embedding_dim
            elif is_transparent_wrapped:
                # Transparent multi_channel modes (onehot/sinusoidal/random/
                # coordinates). The wrapper holds the (N, D) feature matrix; we
                # capture N + the mode so the per-trial factory can rebuild it
                # with a tuned sinusoidal_dim / random_identifier_dim.
                inner_model_cls = type(torch_model.inner)
                model_args = getattr(torch_model.inner, '_args', None)
                _tw_num_stations = int(torch_model.channel_features.shape[0])
                _tw_mode = str(torch_model.mode)
                # From the model instance, NOT config (coordinates/station_ids
                # are never in config) — otherwise the rebuild falls back to
                # integer ids and coordinates raises.
                _tw_station_ids = torch_model.station_ids
                _tw_coords = torch_model._coordinates
                _tw_seed = int(torch_model.random_seed)
            elif is_entity_wrapped:
                inner_model_cls = type(torch_model.inner)
                model_args = getattr(torch_model.inner, '_args', None)
                # Capture EntityWrapper parameters so we can re-wrap per trial todo: these arg names may change.
                _ew_enc_in = self.config.get(
                    'enc_in',
                    torch_model.enc_proj.in_features - torch_model.embedding.embedding_dim,
                )
                _ew_num_embeddings = torch_model.embedding.num_embeddings
                _ew_entity_id_col = torch_model.entity_id_col
            else:
                inner_model_cls = type(torch_model)
                model_args = getattr(torch_model, '_args', None)

            if model_args is None:
                from types import SimpleNamespace

                model_args = SimpleNamespace(**self.config)

            if is_transparent_wrapped or is_pe_transparent:
                # Make the transparent-dim knobs visible on the namespace so a
                # tuned value from the trial config propagates into the rebuild
                # (make_trainable only sets config keys the namespace already has).
                for _attr in ('sinusoidal_dim', 'random_identifier_dim'):
                    if not hasattr(model_args, _attr):
                        setattr(model_args, _attr, self.config.get(_attr, 16))

            if is_pe_transparent:
                # Base (identifier-free) input width: the captured namespace
                # carries the WIDENED enc_in the inner model was built with.
                _pt_base_enc_in = int(getattr(model_args, 'enc_in', self.config.get('enc_in', 1))) - _pt_fixed_dim

            # Build a model factory that rebuilds the full model (including
            # entity / transparent wrapping) from a namespace of args.
            if is_pe_transparent:

                def _model_factory(args):
                    # Tuned dims change the input width: rebuild the inner
                    # model at enc_in = base + dim, then re-wrap.
                    if _pt_mode == 'sinusoidal':
                        dim = int(getattr(args, 'sinusoidal_dim', _pt_fixed_dim))
                    elif _pt_mode == 'random':
                        dim = int(getattr(args, 'random_identifier_dim', _pt_fixed_dim))
                    else:  # onehot / coordinates / numeric_id: width is fixed
                        dim = _pt_fixed_dim
                    args.enc_in = _pt_base_enc_in + dim
                    inner = inner_model_cls(args).float()
                    return EntityTransparentWrapper(
                        inner_model=inner,
                        mode=_pt_mode,
                        station_ids=_pt_station_ids,
                        coordinates=_pt_coords,
                        sinusoidal_dim=int(getattr(args, 'sinusoidal_dim', 16)),
                        random_dim=int(getattr(args, 'random_identifier_dim', 16)),
                        random_seed=_pt_seed,
                    )

            elif is_channel_wrapped:

                def _model_factory(args):
                    inner = inner_model_cls(args).float()
                    emb_size = getattr(args, 'embedding_size', _cw_emb_size)
                    return ChannelEntityWrapper(
                        inner_model=inner,
                        num_stations=_cw_num_stations,
                        embedding_size=emb_size,
                    )

            elif is_entity_wrapped:

                def _model_factory(args):
                    inner = inner_model_cls(args).float()
                    emb_size = getattr(args, 'embedding_size', 10)
                    return EntityWrapper(
                        inner_model=inner,
                        enc_in=_ew_enc_in,
                        num_embeddings=_ew_num_embeddings,
                        embedding_size=emb_size,
                        entity_id_col=_ew_entity_id_col,
                    )

            elif is_transparent_wrapped:

                def _model_factory(args):
                    inner = inner_model_cls(args).float()
                    return ChannelTransparentWrapper(
                        inner_model=inner,
                        mode=_tw_mode,
                        num_stations=_tw_num_stations,
                        sinusoidal_dim=int(
                            getattr(
                                args,
                                'sinusoidal_dim',
                                self.config.get('sinusoidal_dim', 16),
                            )
                        ),
                        random_dim=int(
                            getattr(
                                args,
                                'random_identifier_dim',
                                self.config.get('random_identifier_dim', 16),
                            )
                        ),
                        random_seed=_tw_seed,
                        coordinates=_tw_coords,
                        station_ids=_tw_station_ids,
                    )
            else:
                _model_factory = None

            def _loaders_factory(cfg: dict) -> dict:
                """Rebuild dataset + loaders for a trial that tunes a
                data-layer dim (runs inside the Ray worker; captures
                nothing, so the closure stays small)."""
                from liulian.pipeline import build_dataset, build_loaders

                return build_loaders(build_dataset(cfg), cfg)

            # Per-trial loader rebuilds are only needed when transparent
            # identifiers are injected at the DATA layer; with
            # id_injection='model' the loaders are identifier-agnostic and
            # the wrapper rebuild (above) handles tuned dims.
            _inj_is_data = str(self.config.get('id_injection', 'data')).strip().lower() == 'data'

            trainable = None
            try:
                from liulian.optim.ray_optimizer import make_trainable

                # Determine if checkpoint saving is enabled
                _save_ckpts = True
                if self.optimizer is not None and hasattr(self.optimizer, 'config'):
                    _save_ckpts = self.optimizer.config.get('save_checkpoints', True)

                trainable = make_trainable(
                    model_cls=inner_model_cls,
                    model_args=model_args,
                    loaders=loaders,
                    base_config=self.config,
                    model_factory=_model_factory,
                    save_checkpoints=_save_ckpts,
                    loaders_factory=_loaders_factory if _inj_is_data else None,
                )
            except Exception as exc:
                logger.warning('Could not build HPO trainable: %s', exc)

            hpo_result = self.optimizer.run(
                self.spec,
                search_space,
                trainable,
                loaders=loaders,
            )
            summary['metrics']['hpo'] = {
                'best_config': hpo_result.best_config,
                # Alias for the results.json contract (docs/results_json.md,
                # field `hpo.best_hparams`); `best_config` stays for cli.py.
                'best_hparams': hpo_result.best_config,
                'best_value': hpo_result.best_value,
                'n_trials': hpo_result.n_trials,
                'best_checkpoint_path': hpo_result.best_checkpoint_path,
                'storage_path': hpo_result.storage_path,
            }
            logger.ok(
                'HPO complete — best value: %.6f, config: %s',
                hpo_result.best_value,
                hpo_result.best_config,
            )

            # Persist best hyper-parameters as a stand-alone YAML file
            # so that the result is easily reusable without re-running HPO.
            try:
                import datetime
                import yaml as _yaml

                best_hparams_path = os.path.join(self._artifacts_dir, 'best_hparams.yaml')
                best_hparams_record = {
                    'best_config': hpo_result.best_config,
                    'best_metric_value': float(hpo_result.best_value),
                    'metric_name': str(self.config.get('loss', 'mse')),
                    'n_trials': hpo_result.n_trials,
                    'dataset': str(self.config.get('data', '')),
                    'model': str(self.config.get('model', '')),
                    'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                }
                with open(best_hparams_path, 'w') as fh:
                    _yaml.safe_dump(best_hparams_record, fh, default_flow_style=False)
                logger.info('Best hyper-parameters saved to %s', best_hparams_path)
            except Exception as exc:
                logger.debug('Could not save best_hparams.yaml: %s', exc)

            # POST-HPO FLOW — load-and-evaluate, NO retraining. Steps:
            #   1. best_hparams.yaml was just persisted above (reusable without re-running HPO).
            #   2. Merge the winning hypers over the base config -> best_cfg (next line).
            #   3. If the winner tuned a DATA-layer dim (sinusoidal/random), rebuild
            #      dataset+loaders with that dim and re-sync enc_in (see block below).
            #   4. Build a ForecastTrainer from best_cfg, WITH inverse_transform (this is
            #      where the final denorm metrics become possible).
            #   5. Build a FRESH model with the winning hypers (same factory as the trials).
            #   6. Load the best trial's BEST-EPOCH checkpoint (get_best_checkpoint) into it.
            #   7. Loaded OK -> trainer.evaluate(test) directly (epochs_run=0, incl. denorm_*)
            #      — the weights ARE the best trial's best epoch, so retraining is
            #      unnecessary and would only add variance.
            #   8. Checkpoint missing -> RuntimeError, LOUD (hpo_save_checkpoints must stay
            #      true; the old silent retrain fallback was removed — job 11579994 incident).
            #   9. trainer.predict(test) collects raw preds for viz/.npz, then everything is
            #      written to results.json (metrics.test carries mse/... and denorm_*).
            best_cfg = {**self.config, **hpo_result.best_config}
            best_cfg.pop('search_space', None)

            # DATA-LAYER dim reconciliation for the post-HPO retrain.
            #
            # Background: for id_injection='data' identifier modes (sinusoidal /
            # random), the per-station identity vector is CONCATENATED AS EXTRA
            # INPUT COLUMNS into every window at DATASET BUILD time — so the
            # loader tensors physically contain it, and the model input width is
            # enc_in = base_features + dim. The dim itself (sinusoidal_dim /
            # random_identifier_dim = DATA_LAYER_DIM_KEYS) is an HPO knob.
            #
            # During HPO the driver built ONE set of loaders with the BASE
            # config's dim and shipped it to all trials; a trial that sampled a
            # DIFFERENT dim rebuilt its own loaders inside the worker
            # (loaders_factory in make_trainable). But the driver-side loaders
            # here are still the BASE-dim ones. Example: base sinusoidal_dim=8
            # -> driver windows have 1+8=9 columns; if the WINNING trial sampled
            # dim=17 (windows 1+17=18 columns, model built with enc_in=18), then
            # retraining/evaluating that model on the frozen 9-column loaders
            # would be a shape mismatch (or, worse, a silently wrong input
            # space). So: rebuild dataset+loaders with the winning dim, re-read
            # the ACTUAL feature count from a real batch, and sync enc_in before
            # building the final model. Gated on _inj_is_data because
            # model-layer injection never changes the loader tensors.
            from liulian.optim.search_spaces import DATA_LAYER_DIM_KEYS

            _best_dims = {k: hpo_result.best_config[k] for k in DATA_LAYER_DIM_KEYS if k in hpo_result.best_config}
            _rebuilt_enc_in: int | None = None
            if _best_dims and _inj_is_data:
                from liulian.pipeline import build_dataset, build_loaders

                loaders = build_loaders(build_dataset(best_cfg), best_cfg)
                train_loader = loaders['train']
                val_loader = loaders['val']
                test_loader = loaders.get('test')
                _xb = next(iter(train_loader))[0]
                _rebuilt_enc_in = int(_xb.shape[-1])
                best_cfg['enc_in'] = _rebuilt_enc_in
                logger.info(
                    'Rebuilt loaders for best data-layer dims %s (enc_in=%d)',
                    _best_dims,
                    _rebuilt_enc_in,
                )

            trainer = ForecastTrainer(
                config=best_cfg,
                checkpoint_dir=ckpt_dir,
                exp_logger=self.exp_logger,
                inverse_transform=self._resolve_inverse_transform_fn(loaders),
            )

            # Rebuild model with best hypers. FAIL LOUD on any error: silently
            # keeping the pre-HPO model object (the old fallback) would either
            # crash later at load_state_dict with a confusing shape mismatch or,
            # worse, evaluate a model built with the WRONG hypers while
            # reporting them as the best config.
            try:
                from types import SimpleNamespace as _NS

                best_args = _NS(**{**vars(model_args), **hpo_result.best_config})
                if _rebuilt_enc_in is not None and hasattr(best_args, 'enc_in'):
                    best_args.enc_in = _rebuilt_enc_in
                if _model_factory is not None:
                    torch_model = _model_factory(best_args)
                else:
                    torch_model = inner_model_cls(best_args).float()
            except Exception as exc:
                raise RuntimeError(
                    f'Failed to rebuild the model with the best HPO config '
                    f'{hpo_result.best_config!r}: {type(exc).__name__}: {exc}'
                ) from exc

            # Load the best checkpoint from Ray Tune results
            _loaded_checkpoint = False
            if hpo_result.best_checkpoint_path is not None:
                try:
                    ckpt_dir_path = hpo_result.best_checkpoint_path
                    # Find .pth file in the checkpoint directory
                    # INVARIANT: the trainable writes exactly ONE file per Ray
                    # checkpoint dir, named 'model.pth' (ray_optimizer.py, the
                    # epoch_callback save). Multiple .pth files here would mean
                    # that contract broke and sorted()[0] could silently load
                    # the wrong epoch — fail loud instead of guessing.
                    pth_files = [f for f in os.listdir(ckpt_dir_path) if f.endswith('.pth')]
                    if len(pth_files) > 1:
                        raise RuntimeError(
                            f'Expected exactly one .pth in Ray checkpoint dir '
                            f'{ckpt_dir_path!r}, found {sorted(pth_files)!r} — '
                            f'the one-checkpoint-per-report contract broke.'
                        )
                    if pth_files:
                        import torch as _torch

                        state_path = os.path.join(ckpt_dir_path, pth_files[0])
                        torch_model.load_state_dict(_torch.load(state_path, weights_only=True))
                        torch_model.eval()
                        logger.ok('Loaded best checkpoint from %s', state_path)
                        _loaded_checkpoint = True
                except Exception as exc:
                    logger.warning('Could not load best checkpoint: %s', exc)

            if _loaded_checkpoint:
                # Evaluate the loaded model to populate train_result
                train_result = {
                    'best_val_score': hpo_result.best_value,
                    'epochs_run': 0,
                    'history': [],
                    'metrics': {'training': {}, 'validation': {}, 'test': {}},
                }
                if test_loader is not None:
                    test_metrics = trainer.evaluate(torch_model, test_loader)
                    train_result['metrics']['test'] = test_metrics
            else:
                # No checkpoint available — retrain with best config
                raise RuntimeError(
                    'Best checkpoint not found after HPO.  Checkpoint saving may have been disabled. Please retrain with best config.'
                )
                logger.warning(
                    'No checkpoint available from HPO (checkpoint saving '
                    'may have been disabled). Retraining with best config.'
                )
                train_result = trainer.fit(
                    torch_model,
                    train_loader,
                    val_loader,
                    test_loader,
                )
            train_metrics = train_result.get('metrics', {})
            summary['metrics']['training'] = train_metrics.get('training', {})
            summary['metrics']['validation'] = train_metrics.get('validation', {})
            summary['metrics']['test'] = train_metrics.get('test', {})
            summary['metrics']['history'] = train_result.get('history', [])
            summary['metrics']['best_val_score'] = train_result.get('best_val_score')
            summary['metrics']['best_epoch'] = train_result.get('best_epoch')
            summary['metrics']['epochs_run'] = train_result.get('epochs_run', 0)

            self._sm.transition(LifecycleState.EVAL)
            self._fire('on_eval_end', metrics=train_metrics.get('test', {}))

            # # Compute task metrics + predictions on the retrained model
            # self._compute_task_metrics(
            #     summary, torch_model, trainer
            # )  # todo: is this necessary if trainer.evaluate is already done?
            if test_loader is not None:
                pred_result = trainer.predict(torch_model, test_loader)
                summary['predictions'] = pred_result
            return

        # ----- Standard single-trial path -----

        trainer = ForecastTrainer(
            config=self.config,
            checkpoint_dir=ckpt_dir,
            exp_logger=self.exp_logger,
            inverse_transform=self._resolve_inverse_transform_fn(loaders),
        )

        if train and train_loader is not None and val_loader is not None:
            self._sm.transition(LifecycleState.TRAIN)
            train_result = trainer.fit(torch_model, train_loader, val_loader, test_loader)
            train_metrics = train_result.get('metrics', {})
            summary['metrics']['training'] = train_metrics.get('training', {})
            summary['metrics']['validation'] = train_metrics.get('validation', {})
            summary['metrics']['test'] = train_metrics.get('test', {})
            summary['metrics']['history'] = train_result.get('history', [])
            summary['metrics']['best_val_score'] = train_result.get('best_val_score')
            summary['metrics']['best_epoch'] = train_result.get('best_epoch')
            summary['metrics']['epochs_run'] = train_result.get('epochs_run', 0)

            self._fire(
                'on_epoch_end',
                epoch=train_result['epochs_run'],
                loss=train_result['history'][-1]['train_loss'],
            )

            # Transition to EVAL
            self._sm.transition(LifecycleState.EVAL)
            self._fire('on_eval_end', metrics=train_metrics.get('test', {}))

        elif eval and not train:
            # Eval-only: load checkpoint and evaluate
            self._sm.transition(LifecycleState.TRAIN)
            self._sm.transition(LifecycleState.EVAL)

            import torch as _torch

            ckpt_path = os.path.join(ckpt_dir, 'checkpoint')
            if os.path.exists(ckpt_path):
                torch_model.load_state_dict(_torch.load(ckpt_path, map_location=trainer.device, weights_only=True))
                logger.ok('Loaded checkpoint: %s', ckpt_path)

            if test_loader is not None:
                test_metrics = trainer.evaluate(torch_model, test_loader)
                summary['metrics']['test'] = test_metrics
                self._fire('on_eval_end', metrics=test_metrics)
                logger.info('Test metrics: %s', test_metrics)

        # # Compute liulian task-level metrics on a sample
        # self._compute_task_metrics(summary, torch_model, trainer)

        # Collect raw predictions for downstream visualization & .npz export.
        # This is NOT redundant with per-epoch evaluate()—evaluate() only
        # returns aggregate metrics; predict() returns raw (preds, trues, times)
        # tensors needed by the viz pipeline.
        if test_loader is not None:
            pred_result = trainer.predict(torch_model, test_loader)
            summary['predictions'] = pred_result
            logger.info(
                'Predictions collected: preds=%s, trues=%s, times=%s',
                list(pred_result['preds'].shape),
                list(pred_result['trues'].shape),
                list(pred_result['times'].shape),
            )

        if infer:
            if self._sm.can_transition(LifecycleState.INFER):
                self._sm.transition(LifecycleState.INFER)
            self._fire('on_infer_complete')

    def _compute_task_metrics(
        self,
        summary: Dict[str, Any],
        torch_model: Any,
        trainer: Any,
    ) -> None:
        """Compute liulian PredictionTask metrics on a sample batch."""
        try:
            import torch as _torch

            test_split = self.dataset.get_split('test')
            X_sample, y_sample = test_split.get_batch(
                batch_size=32
            )  # todo: this is computed on a given sample batch. Maybe remove it.

            cfg = self.config
            pred_len = cfg.get('pred_len', 1)
            seq_len = cfg.get('seq_len', X_sample.shape[1])

            # For forecast windows: X_sample and y_sample span seq_len+pred_len.
            # Slice to match model expectations.
            x_enc = X_sample[:, :seq_len, :]  # encoder input
            y_true = y_sample[:, -pred_len:, :]  # ground truth for loss

            batch = self.task.prepare_batch(
                {
                    'X': x_enc,
                    'y': y_true,
                }
            )  # these are numpy arrays.

            x_tensor = _torch.tensor(batch['X'], dtype=_torch.float32).to(
                trainer.device
            )  # todo: can this just use x_enc?

            torch_model.eval()
            with _torch.no_grad():
                dec = _torch.zeros(
                    x_tensor.size(0),
                    pred_len,
                    x_tensor.size(2),
                    device=trainer.device,
                )
                mark = _torch.zeros(
                    x_tensor.size(0),
                    x_tensor.size(1),
                    1,
                    device=trainer.device,
                )  # todo: maybe this is known or useful for some tasks?
                mark_dec = _torch.zeros(
                    x_tensor.size(0),
                    pred_len,
                    1,
                    device=trainer.device,
                )
                pred = torch_model(x_tensor, mark, dec, mark_dec)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = pred[:, -pred_len:, :]

            model_output = {'predictions': pred.cpu().numpy()}
            task_metrics = self.task.compute_metrics(model_output, batch)
            summary['metrics']['task_metrics'] = task_metrics
            logger.info('PredictionTask metrics: %s', task_metrics)
        except Exception as exc:
            logger.warning('Could not compute task metrics: %s', exc)

    # ------------------------------------------------------------------
    # Simple (non-torch) path
    # ------------------------------------------------------------------

    def _run_simple(
        self,
        summary: Dict[str, Any],
        *,
        train: bool,
        eval: bool,
        infer: bool,
        batch_size: int,
    ) -> None:
        """Drive the lifecycle with single-batch forward passes."""
        if train:
            self._sm.transition(LifecycleState.TRAIN)
            train_split = self.dataset.get_split('train')
            X, y = train_split.get_batch(batch_size=batch_size)
            batch = self.task.prepare_batch({'X': X, 'y': y})
            output = self.model.forward(batch)
            loss = self.task.build_loss(output, batch)

            if self.exp_logger:
                self.exp_logger.log_metrics(step=1, metrics={'train_loss': float(loss)})
            summary['metrics']['train_loss'] = float(loss)
            self._fire('on_epoch_end', epoch=1, loss=loss)
            logger.info('Train loss: %.6f', loss)

        if eval and train:
            self._sm.transition(LifecycleState.EVAL)
            val_split = self.dataset.get_split('val')
            X, y = val_split.get_batch(batch_size=batch_size)
            batch = self.task.prepare_batch({'X': X, 'y': y})
            output = self.model.forward(batch)
            metrics = self.task.compute_metrics(output, batch)

            if self.exp_logger:
                self.exp_logger.log_metrics(step=1, metrics=metrics)
            summary['metrics'].update(metrics)
            self._fire('on_eval_end', metrics=metrics)
            logger.info('Eval metrics: %s', metrics)

        if infer:
            if self._sm.state == LifecycleState.EVAL:
                self._sm.transition(LifecycleState.INFER)
            elif self._sm.state == LifecycleState.TRAIN:
                self._sm.transition(LifecycleState.EVAL)
                self._sm.transition(LifecycleState.INFER)
            self._fire('on_infer_complete')
            logger.ok('Inference phase completed')

    # ------------------------------------------------------------------
    # Pause / Resume
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Pause the experiment (only valid during TRAIN).

        Raises:
            ValueError: If the current state does not allow pausing.
        """
        self._sm.transition(LifecycleState.PAUSED)
        logger.info('Experiment paused.')

    def resume(self) -> None:
        """Resume a paused experiment back to TRAIN.

        Raises:
            ValueError: If the experiment is not in PAUSED state.
        """
        self._sm.transition(LifecycleState.TRAIN)
        logger.info('Experiment resumed.')

    @property
    def state(self) -> LifecycleState:
        """Current lifecycle state of the experiment."""
        return self._sm.state

    @property
    def artifacts_dir(self) -> Optional[str]:
        """Path to the artifacts directory (set after :meth:`run`)."""
        return self._artifacts_dir

    # ------------------------------------------------------------------
    # Logger auto-creation
    # ------------------------------------------------------------------

    def _maybe_create_logger(self) -> LoggerInterface:
        """Create a logger from config keys or fall back to local.

        Recognised config keys:

        * ``wandb_project`` — enables WandB logging.
        * ``wandb_entity``  — optional WandB team/user.
        * ``dev_run``       — when *True*, disables wandb even if project
          is set (matches swiss-river reference convention).

        Returns:
            A :class:`LoggerInterface` instance.
        """
        wandb_project = self.config.get('wandb_project')
        dev_run = self.config.get('dev_run', False)

        if wandb_project and not dev_run:
            try:
                from liulian.loggers.wandb_logger import WandbLogger

                return WandbLogger(
                    project=wandb_project,
                    entity=self.config.get('wandb_entity'),
                    config=self.config,
                    run_dir=self._artifacts_dir or 'artifacts/logs',
                )
            except Exception as exc:
                logger.warning(
                    'Could not create WandbLogger (%s); falling back to local.',
                    exc,
                )

        from liulian.loggers.local_logger import LocalFileLogger

        return LocalFileLogger(run_dir=self._artifacts_dir or 'artifacts/logs')
