"""Training lifecycle around the authors' GeRaFStage1 loss, for both datasets.

All target preprocessing is part of explicit execution, never metadata planning.
Historical local GeRaF caches/checkpoints have different semantics and cannot
be resumed here. No fitted gain, peak scaling, geometry prior or mask repair is
inserted into the source training objective.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from rift.geraf_source import (SCHEMA, build_model, recipe_from_config, recipe_for_data, sample_frame,
                               source_step, predict_native, fixed_numpy_seed, WARM_START_VIEWS)
from rift.geraf_source_data import identity_hash
from rift.gotcha_training import atomic_json, atomic_save
from rift.vendor.geraf_sens.scheduler import IterLRScheduler

TARGET_SCHEMA = SCHEMA + '_targets_lazy_v1'


class InterruptedPreparation(Exception):
    pass


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def lattice_points(indices, n, radius, device):
    i = torch.as_tensor(indices, device=device, dtype=torch.int64)
    xyz = torch.stack((i // n**2, (i // n) % n, i % n), -1).double()
    return (2 * xyz / (n - 1) - 1) * radius


def measured_values(acquisition, response, xyz, trans_power):
    with torch.no_grad():
        values = acquisition.matched_filter(response, xyz).abs() / trans_power
        if not bool(torch.isfinite(values).all()):
            raise ValueError('Nonfinite measured MF target')
        # Preserve the old dense writer's FP32 quantization BEFORE interpolation
        # or accumulation; evaluating MF directly at the query is not equivalent.
        result = values.cpu().float().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Measured MF target overflows float32')
        return result


class LazyMFTarget:
    """One response and on-demand lattice corners; never a per-view cube/file."""
    def __init__(self, acquisition, response, recipe, device, stopped):
        self.acquisition, self.response, self.recipe = acquisition, response, recipe
        self.device, self.stopped = device, stopped

    def sample(self, normalized_points):
        from rift.geraf_target_sampling import sample_lattice
        r = self.recipe
        def fetch(indices):
            values = np.empty(len(indices), dtype=np.float32)
            for start in range(0, len(indices), r['point_chunk']):
                if self.stopped():
                    raise InterruptedPreparation()
                ids = indices[start:start+r['point_chunk']]
                xyz = lattice_points(ids, r['mf_grid'], r['extent_m'], self.device)
                values[start:start+len(ids)] = measured_values(
                    self.acquisition, self.response, xyz, r['trans_power'])
            return values
        if self.stopped():
            raise InterruptedPreparation()
        return sample_lattice(normalized_points, r['mf_grid'], fetch)


class SourceTargets:
    """Lazy per-view targets and one resumable accumulated lattice per head."""
    def __init__(self, root, data, recipe):
        if recipe.get('target_storage') != 'lazy_trilinear_accumulated_only_v1':
            raise ValueError('GeRaF targets require the lazy-storage recipe; use a new run/cache identity')
        self.root, self.data, self.recipe = Path(root), data, recipe
        self.identity = dict(schema=TARGET_SCHEMA, contract=data.contract, recipe=recipe)
        self.digest = identity_hash(self.identity)
        self.verified = {}

    def start(self):
        manifest = self.root / 'source_targets.json'
        if self.root.exists() and any(self.root.iterdir()):
            if not manifest.is_file() or json.loads(manifest.read_text()) != self.identity:
                raise ValueError('Source GeRaF target cache identity mismatch; historical targets cannot be reused')
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest, self.identity)

    def grid_chunks(self, device):
        n, radius = self.recipe['mf_grid'], self.recipe['extent_m']
        # Source grid_sample(align_corners=True) maps cube endpoints to endpoints.
        for start in range(0, n**3, self.recipe['point_chunk']):
            i = np.arange(start, min(start + self.recipe['point_chunk'], n**3), dtype=np.int64)
            yield start, lattice_points(i, n, radius, device)

    def _read(self, name):
        path, meta = self.root / (name + '.npy'), self.root / (name + '.json')
        if not path.exists() or not meta.exists():
            return None
        record = json.loads(meta.read_text())
        if record.get('identity') != self.digest or record.get('name') != name:
            raise ValueError('GeRaF MF volume has a different source/role/recipe identity')
        stamp = (path.stat().st_size, path.stat().st_mtime_ns, record.get('sha256'))
        if self.verified.get(name) != stamp:
            if file_hash(path) != record.get('sha256'):
                raise ValueError('GeRaF MF volume checksum mismatch')
            self.verified[name] = stamp
        volume = np.load(path, mmap_mode='r', allow_pickle=False)
        if volume.shape != (self.recipe['mf_grid'],) * 3 or volume.dtype != np.float32:
            raise ValueError('GeRaF MF volume shape/dtype mismatch')
        return volume

    def _finish(self, name, temporary):
        path = self.root / (name + '.npy')
        os.replace(temporary, path)
        atomic_json(self.root / (name + '.json'), dict(identity=self.digest, name=name, sha256=file_hash(path)))

    def get(self, role, view, head, device, stopped):
        self.data.key(role, view, head)  # Role gate precedes all response access.
        if stopped():
            raise InterruptedPreparation()
        acquisition = self.data.acquisition(role, view, head, self.recipe, device)
        response = self.data.response(role, view, head, device)
        return LazyMFTarget(acquisition, response, self.recipe, device, stopped)

    def _partial_path(self, name):
        return self.root / (name + '.partial.pt')

    def _load_partial(self, name, train_keys):
        n = self.recipe['mf_grid']
        path = self._partial_path(name)
        if not path.exists():
            return 0, np.zeros((n, n, n), dtype=np.float32)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (not isinstance(saved, dict) or saved.get('identity') != self.digest
                or saved.get('name') != name or saved.get('train_keys') != train_keys):
            raise ValueError('GeRaF partial accumulation identity/role order mismatch')
        cursor, volume = saved.get('completed_views'), saved.get('volume')
        if (type(cursor) is not int or not 0 <= cursor <= len(train_keys)
                or not isinstance(volume, torch.Tensor) or volume.dtype != torch.float32
                or tuple(volume.shape) != (n, n, n) or not bool(torch.isfinite(volume).all())):
            raise ValueError('Invalid GeRaF partial accumulation cursor/tensor')
        value = volume.numpy().copy()
        if hashlib.sha256(value.tobytes()).hexdigest() != saved.get('sha256'):
            raise ValueError('GeRaF partial accumulation checksum mismatch')
        return cursor, value

    def _save_partial(self, name, train_keys, cursor, volume):
        atomic_save(self._partial_path(name), dict(identity=self.digest, name=name,
            train_keys=train_keys, completed_views=cursor,
            sha256=hashlib.sha256(volume.tobytes()).hexdigest(), volume=torch.from_numpy(volume)))

    def accumulated(self, head, device, stopped):
        if head not in self.data.heads:
            raise PermissionError('Unregistered GeRaF target head')
        name = 'accumulated_train_' + head
        value = self._read(name)
        if value is not None:
            self._partial_path(name).unlink(missing_ok=True)
            return value
        # Pinned docs/PrepareData.md specifies a sum of per-frame MF volumes.
        # Our native-acquisition adapter sums measured magnitudes on the same
        # world grid, restricted to training roles. No deposition,
        # coverage weighting, prediction history, normalization or thresholding.
        views = self.data.views('train')
        train_keys = [self.data.key('train', view, head) for view in views]
        cursor, volume = self._load_partial(name, train_keys)
        temp = self.root / (name + f'.tmp.{os.getpid()}.npy')
        try:
            for index in range(cursor, len(views)):
                if stopped():
                    raise InterruptedPreparation()
                target = self.get('train', views[index], head, device, stopped)
                for start, xyz in self.grid_chunks(device):
                    if stopped():
                        raise InterruptedPreparation()
                    values = measured_values(target.acquisition, target.response, xyz,
                                             self.recipe['trans_power'])
                    block = volume.reshape(-1)[start:start+len(xyz)]
                    # Same per-voxel FP32 summation order; only a chunk is live.
                    np.add(block, values, out=block)
                    if not np.isfinite(block).all():
                        raise ValueError('Nonfinite accumulated MF target')
                # Commit only complete views. A partial next view can be
                # recomputed without duplicating any already committed sum.
                self._save_partial(name, train_keys, index + 1, volume)
            with temp.open('wb') as f:
                np.save(f, volume, allow_pickle=False)
            self._finish(name, temp)
            self._partial_path(name).unlink(missing_ok=True)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        return self._read(name)


def make_optimizer(models, recipe):
    sdf, other = [], []
    for model in models.values():
        for name, param in model.named_parameters():
            (sdf if name.startswith('sdf_network.') else other).append(param)
    optimizer = torch.optim.AdamW([dict(params=sdf, lr=recipe['sdf_lr']),
                                  dict(params=other, lr=recipe['other_lr'])],
                                 lr=recipe['other_lr'], weight_decay=recipe['weight_decay'])
    scheduler = IterLRScheduler(optimizer, [dict(type='CosineAnnealingLR',
        eta_min=recipe['eta_min'], by_epoch=False, begin=0)], recipe['steps'])
    return optimizer, scheduler


def evaluate(models, data, recipe, targets, device, stopped):
    totals = dict(mf_sse=0., mf_energy=0., mf_count=0, native_sse=0., native_energy=0., native_count=0)
    for head in data.heads:
        model = models[head]
        for index, view in enumerate(data.views('validation')):
            if stopped():
                return None
            volume = targets.get('validation', view, head, device, stopped)
            acquisition = data.acquisition('validation', view, head, recipe, device)
            with fixed_numpy_seed(recipe['seed'] + index):
                frame = sample_frame(acquisition, recipe, data.key('validation', view, head), volume, volume)
            prediction, inside = predict_native(model, frame, acquisition, return_inbounds=True)
            query = frame['tgt_sampled_poses_glb'].reshape(-1, 3)[inside]
            if not len(query):
                raise ValueError('Source GeRaF interpolation has no inbounds validation queries')
            with torch.no_grad():
                predicted_mf = acquisition.matched_filter(prediction, query).abs()
                measured_mf = frame['mf_sampled_value'].flatten()[inside].double()
                if not bool(torch.isfinite(prediction).all() & torch.isfinite(predicted_mf).all()):
                    raise ValueError('Nonfinite GeRaF validation prediction')
                totals['mf_sse'] += float((predicted_mf - measured_mf).square().sum())
                totals['mf_energy'] += float(measured_mf.square().sum())
                totals['mf_count'] += measured_mf.numel()
                measured = data.response('validation', view, head, device) / recipe['trans_power']
                totals['native_sse'] += float((prediction - measured).abs().square().sum())
                totals['native_energy'] += float(measured.abs().square().sum())
                totals['native_count'] += measured.numel()
    if not totals['mf_count'] or totals['mf_energy'] <= 0 or totals['native_energy'] <= 0:
        raise ValueError('Cannot select a GeRaF checkpoint without nonzero validation targets')
    return dict(**totals, mf_magnitude_mse=totals['mf_sse'] / totals['mf_count'],
                mf_relative_mse=totals['mf_sse'] / totals['mf_energy'],
                native_complex_relative_mse=totals['native_sse'] / totals['native_energy'],
                selection_metric='mf_magnitude_mse', masked=False)


def light_power_warm_start(models, data, recipe, targets, accumulated, device, stopped):
    """Configured start of light_power: least-squares scale of the initial render to the TRAIN MF targets.

    Uses the validation readout path (no bank updates) on the first WARM_START_VIEWS
    training views, with fixed sampling seeds so the training RNG stream is untouched.
    The render is linear in exp(light_power), so one closed-form ratio sets the start.
    """
    record = {}
    for head, model in models.items():
        numerator = denominator = 0.0
        views = data.views('train')[:WARM_START_VIEWS]
        for index, view in enumerate(views):
            if stopped():
                return None
            volume = targets.get('train', view, head, device, stopped)
            acquisition = data.acquisition('train', view, head, recipe, device)
            with fixed_numpy_seed(recipe['seed'] + index):
                frame = sample_frame(acquisition, recipe, data.key('train', view, head), volume, accumulated[head])
            prediction, inside = predict_native(model, frame, acquisition, return_inbounds=True)
            query = frame['tgt_sampled_poses_glb'].reshape(-1, 3)[inside]
            if not len(query):
                continue
            with torch.no_grad():
                predicted = acquisition.matched_filter(prediction, query).abs()
                measured = frame['mf_sampled_value'].flatten()[inside].double()
                numerator += float((predicted * measured).sum())
                denominator += float(predicted.square().sum())
        scale = numerator / denominator if denominator > 0 else float('nan')
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('GeRaF light_power warm start is degenerate')
        with torch.no_grad():
            start = model.signal_network.light_power.add_(float(np.log(scale)))
        record[head] = dict(views=len(views), scale=scale, light_power=float(start))
    return record


def validate_checkpoint(checkpoint, data, recipe):
    if (not isinstance(checkpoint, dict) or checkpoint.get('schema') != SCHEMA + '_checkpoint'
            or checkpoint.get('data_identity') != data.identity
            or checkpoint.get('contract') != data.contract or checkpoint.get('recipe') != recipe):
        raise ValueError('Source GeRaF checkpoint source/object/roles/recipe mismatch')
    step = checkpoint.get('step')
    if type(step) is not int or not 0 <= step <= recipe['steps']:
        raise ValueError('Invalid GeRaF optimizer step')
    train = data.views('train')
    expected = {data.key('train', v, h): step // len(train) + (i < step % len(train))
                for h in data.heads for i, v in enumerate(train)}
    if checkpoint.get('exposures') != expected:
        raise ValueError('GeRaF checkpoint exposure counts disagree with optimizer steps')
    if set(checkpoint.get('models', {})) != set(data.heads):
        raise ValueError('GeRaF checkpoint polarization heads changed')
    expected_target_identity = identity_hash(dict(schema=TARGET_SCHEMA, contract=data.contract, recipe=recipe))
    if checkpoint.get('target_identity') != expected_target_identity:
        raise ValueError('GeRaF checkpoint target preprocessing identity changed')
    for head, state in checkpoint['models'].items():
        used = {key for key in expected if expected[key] > 0 and key in
                {data.key('train', v, head) for v in train}}
        for key in ('ant_real_list', 'ant_imag_list', 'ant_chunk_id'):
            if set(state.get(key, {})) != used:
                raise ValueError('GeRaF source bank role/coverage mismatch')
        for key, pointer in state['ant_chunk_id'].items():
            if pointer != expected[key] % recipe['bank_size']:
                raise ValueError('GeRaF source bank pointer disagrees with exposures')
        for view in train:
            key = data.key('train', view, head)
            if key not in used:
                continue
            acquisition = data.acquisition('train', view, head, recipe, 'cpu')
            for field in ('ant_real_list', 'ant_imag_list'):
                groups = state[field][key]
                if len(groups) != recipe['bank_size']:
                    raise ValueError('GeRaF cached antenna bank size changed')
                for group, tensor in enumerate(groups):
                    expected_shape = (len(range(group, len(acquisition.tx), recipe['bank_size'])), len(acquisition.frequencies))
                    if (not torch.is_tensor(tensor) or tuple(tensor.shape) != expected_shape
                            or tensor.is_complex() or not bool(torch.isfinite(tensor).all())):
                        raise ValueError('GeRaF cached native antenna tensor is invalid')
    history = checkpoint.get('validation_history')
    if not isinstance(history, list) or any(type(row.get('step')) is not int or
            not 0 < row['step'] <= step or not math_isfinite_nonnegative(row.get('mf_magnitude_mse'))
            for row in history):
        raise ValueError('Invalid GeRaF validation history')
    if [r['step'] for r in history] != sorted(set(r['step'] for r in history)):
        raise ValueError('GeRaF validation steps must be strictly increasing')
    best = min((r['mf_magnitude_mse'] for r in history), default=None)
    if checkpoint.get('best_mse') != best:
        raise ValueError('GeRaF best metric differs from validation history')
    pending = checkpoint.get('pending_validation')
    if pending not in (None, step) or (pending is not None and any(r['step'] == step for r in history)):
        raise ValueError('Invalid GeRaF pending validation')
    complete = step == recipe['steps'] and bool(history) and history[-1]['step'] == step and pending is None
    if checkpoint.get('complete') is not complete:
        raise ValueError('GeRaF completion requires terminal unmasked validation')


def math_isfinite_nonnegative(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value) and value >= 0


def load_selected_models(checkpoint, data, device='cpu'):
    """Bind readout to the same unmasked-MF-selected checkpoint, before reads."""
    from rift.geraf_source import DEFAULTS, LEGACY_LIGHT_POWER_START, LEGACY_RECEIVER_GEOMETRY
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('recipe'), dict):
        raise ValueError('Source GeRaF checkpoint lacks a recipe')
    saved = checkpoint['recipe']
    legacy = dict(receiver_geometry=LEGACY_RECEIVER_GEOMETRY, light_power_start=LEGACY_LIGHT_POWER_START)
    config = {k: saved[k] for k in DEFAULTS if k not in legacy}
    config.update({k: saved.get(k, v) for k, v in legacy.items()})
    recipe = recipe_for_data(config, data)
    validate_checkpoint(checkpoint, data, recipe)
    history = checkpoint['validation_history']
    if not history:
        raise ValueError('GeRaF checkpoint has no validation selection')
    selected = min(history, key=lambda row: row['mf_magnitude_mse'])
    if selected['step'] != checkpoint['step'] or selected.get('masked') is not False:
        raise ValueError('GeRaF readout requires the unmasked validation-selected checkpoint')
    models = {head: build_model(recipe, device) for head in data.heads}
    for head, model in models.items():
        model.load_state_dict(checkpoint['models'][head], strict=True)
        model.update_step(source_step(recipe, checkpoint['step']))
        model.eval()
    return models, selected


def train(*, data, output_dir, config, device='cuda', resume=None, cache_root=None, prepare_only=False):
    recipe = recipe_for_data(config, data)
    device, output = torch.device(device), Path(output_dir)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    latest = output / 'checkpoint_latest.pth.tar'
    resume_path = latest if resume == 'auto' and latest.is_file() else None if resume in (None, 'auto') else Path(resume)
    saved = torch.load(resume_path, map_location='cpu', weights_only=False) if resume_path else None
    if saved is not None:
        validate_checkpoint(saved, data, recipe)  # before response/cache payloads
    run_identity = dict(schema=SCHEMA, contract=data.contract, recipe=recipe)
    manifest = output / 'source_run.json'
    if output.exists() and any(output.iterdir()):
        if not manifest.is_file() or json.loads(manifest.read_text()) != run_identity:
            raise ValueError('Nonempty GeRaF output belongs to a different source/recipe')
        if saved is None and not prepare_only and latest.exists():
            raise ValueError('Existing GeRaF checkpoint requires resume')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest, run_identity)
    targets = SourceTargets(cache_root or output / 'source_targets', data, recipe)
    targets.start()
    print(json.dumps(dict(implementation='source_v1', upstream_commit=recipe['upstream_commit'],
        target_grid=recipe['mf_grid'], target_storage=recipe['target_storage'],
        target_bytes_per_head=4*recipe['mf_grid']**3, persistent_per_view_volumes=0,
        training_views=len(data.views('train')), validation_views=len(data.views('validation')),
        historical_v1_recipe_recovered=False)), flush=True)
    stopped = [False]
    def request_stop(signum, frame):
        stopped[0] = True
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        random.seed(recipe['seed']); np.random.seed(recipe['seed']); torch.manual_seed(recipe['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(recipe['seed'])
        models = {head: build_model(recipe, device) for head in data.heads}
        optimizer, scheduler = make_optimizer(models, recipe)
        step, history, best, pending = 0, [], None, None
        exposures = {data.key('train', v, h): 0 for h in data.heads for v in data.views('train')}
        if saved is not None:
            for head in data.heads:
                models[head].load_state_dict(saved['models'][head], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            step, history, best, pending = saved['step'], saved['validation_history'], saved['best_mse'], saved['pending_validation']
            exposures = saved['exposures']
            np.random.set_state(saved['rng_numpy']); random.setstate(saved['rng_python'])
            torch.set_rng_state(saved['rng_torch'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(saved['rng_cuda'])
            scheduler.step(step)
        for model in models.values():
            model.update_step(source_step(recipe, step))
        def save(name, complete=False):
            payload = dict(schema=SCHEMA + '_checkpoint', data_identity=data.identity,
                contract=data.contract, recipe=recipe, models={h: m.state_dict() for h,m in models.items()},
                optimizer=optimizer.state_dict(), step=step, exposures=exposures,
                validation_history=history, best_mse=best, pending_validation=pending,
                complete=complete, rng_numpy=np.random.get_state(), rng_python=random.getstate(),
                rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                target_identity=targets.digest)
            atomic_save(output / name, payload)
        def finish_validation():
            nonlocal pending, best
            metric = evaluate(models, data, recipe, targets, device, lambda: stopped[0])
            if metric is None:
                return False
            history.append(dict(step=step, **metric))
            improved = best is None or metric['mf_magnitude_mse'] < best
            best = metric['mf_magnitude_mse'] if improved else best
            pending = None
            if improved:
                save('checkpoint_best.pth.tar', complete=step == recipe['steps'])
            save('checkpoint_latest.pth.tar', complete=step == recipe['steps'])
            atomic_json(output / 'validation.json', history)
            return True
        try:
            accumulated = {h: targets.accumulated(h, device, lambda: stopped[0]) for h in data.heads}
            if prepare_only:
                return dict(status='prepared', implementation='source_v1', output_dir=str(output),
                            target_storage=recipe['target_storage'], persistent_per_view_volumes=0)
            if saved is None and recipe.get('light_power_start') == 'train_warm_start_16_views':
                warm = light_power_warm_start(models, data, recipe, targets, accumulated, device, lambda: stopped[0])
                if warm is None:
                    return dict(status='interrupted', step=step)
                atomic_json(output / 'light_power_warm_start.json', warm)
                print(json.dumps(dict(light_power_warm_start=warm)), flush=True)
            if pending is not None and not finish_validation():
                save('checkpoint_latest.pth.tar')
                return dict(status='interrupted', step=step)
            if step == recipe['steps']:
                return dict(status='complete', step=step, best_mse=best)
            start_time = time.monotonic()
            while step < recipe['steps']:
                if stopped[0]:
                    save('checkpoint_latest.pth.tar')
                    return dict(status='interrupted', step=step)
                view = data.views('train')[step % len(data.views('train'))]
                optimizer.zero_grad(set_to_none=True)
                log = {}
                for head, model in models.items():
                    model.update_step(source_step(recipe, step))
                    acquisition = data.acquisition('train', view, head, recipe, device)
                    if len(acquisition.tx) < recipe['bank_size']:
                        raise ValueError('A native viewpoint has fewer channels than the source bank size')
                    volume = targets.get('train', view, head, device, lambda: stopped[0])
                    name = data.key('train', view, head)
                    frame = sample_frame(acquisition, recipe, name, volume, accumulated[head])
                    if not bool(frame['loss_mask'].any()):
                        raise ValueError('Source mask removes all rays; no replacement rays or loss are invented')
                    model.radar_cfg = {'native': acquisition}
                    losses = model.loss(frame)  # entire released loss body, including bank updates
                    loss = sum(losses.values())
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('Nonfinite source GeRaF objective')
                    loss.backward()
                    log[head] = {k: float(v.detach()) for k,v in losses.items()}
                    log[head]['retained_ray_fraction'] = float(frame['loss_mask'].float().mean())
                for model in models.values():
                    torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['gradient_clip'])
                optimizer.step()
                step += 1
                scheduler.step(step)
                for head in data.heads:
                    exposures[data.key('train', view, head)] += 1
                for model in models.values():
                    model.update_step(source_step(recipe, step))
                if step % recipe['validation_every'] == 0 or step == recipe['steps']:
                    pending = step
                if step % recipe['checkpoint_every'] == 0 or pending is not None or stopped[0]:
                    save('checkpoint_latest.pth.tar')
                if step == 1 or step % recipe['log_every'] == 0 or step == recipe['steps']:
                    print(json.dumps(dict(step=step, elapsed_seconds=time.monotonic()-start_time, losses=log)), flush=True)
                if pending is not None and not finish_validation():
                    save('checkpoint_latest.pth.tar')
                    return dict(status='interrupted', step=step)
            return dict(status='complete', step=step, best_mse=best, output_dir=str(output))
        except InterruptedPreparation:
            # If no complete update exists, caches alone are resumable. A
            # partially processed multi-head step must not be checkpointed.
            return dict(status='interrupted', step=step, checkpoint=str(latest) if latest.exists() else None)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
