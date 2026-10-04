"""
Video Classification Pipeline Verification
==========================================

Checks that the video configuration application and all four meta-learning
variants actually work, at a budget small enough to run on a laptop in a few
minutes. Run this after changing the TSN implementation, the design space, the
GA, the application, or any of the four approaches.

It runs against the generated dataset
(:mod:`video_configuration.synthetic`), which needs no download and is
produced in the same on-disk layout as UCF101 - so the code path exercised
here is the one a real dataset would use. The generated task is four shapes on
a noisy background and is deliberately easy: these checks establish that the
implementation works, not that it is competitive.

What it checks, and why each check is here:

  1. Design space    - designs round trip through the fixed-length encoding,
                       and arbitrary vectors still decode to buildable designs
                       (the meta-learner's output is unconstrained regression)
  2. Dataset         - the generated frames and list files are consistent, and
                       the loader yields correctly shaped segment stacks
  3. Keyframes       - all five extraction strategies return in-range,
                       1-based indices of the right count, including for
                       videos shorter than their segment count
  4. Model building  - every base architecture and every consensus function
                       builds, runs forward, and produces gradients. Three of
                       the five consensus functions returned None in the
                       recovered code, and the legacy autograd Function it
                       used raised on any current PyTorch
  5. Persistent state - accuracy accumulates across timesteps, and a design
                       change transfers weights instead of starting over
  6. Meta-features   - the vector length is identical at every timestep
                       (ragged vectors cannot be fitted by a meta-learner)
  7. Approaches      - all four run end to end and report a test performance

Usage:
    python experiments/test/verify_video_pipeline.py
    python experiments/test/verify_video_pipeline.py --skip-approaches
    python experiments/test/verify_video_pipeline.py --skip-architectures
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import time

import numpy as np
import torch

from metalearner import load_approach
from applications.configuration.video.video_application import (
    VideoConfigurationApplication, BASE_ARCHITECTURES, CONSENSUS_FUNCTIONS,
    MODEL_ZOO_ARCHITECTURES, model_zoo_available,
)
from video_configuration.keyframes import KEYFRAME_STRATEGIES, sample_indices
from video_configuration.models import TSN
from video_configuration.run import chromosome_to_design, design_to_chromosome


class CheckFailed(Exception):
    """A verification check failed."""


def check(condition, message):
    """Assert a condition with a readable failure."""
    if not condition:
        raise CheckFailed(message)


def make_app(videos_per_class=8, batch_size=4, train_batches=None):
    """Build an application at the verification budget."""
    return VideoConfigurationApplication(
        'synthetic',
        random_seed=42,
        batch_size=batch_size,
        num_workers=0,
        max_train_batches_per_timestep=train_batches,
        candidate_train_batches=2,
        candidate_val_batches=1,
        synthetic_classes=4,
        synthetic_videos_per_class=videos_per_class,
        synthetic_frames=16,
    )


def verify_design_space(app):
    """Designs must survive the encode/decode round trip, and noise must decode."""
    print("\n[1/7] Design space")

    space = app.get_structured_design_space()
    rng = np.random.default_rng(0)
    print(f"  encoding length: {space.encoding_length}")
    print(f"  base architectures: "
          f"{[BASE_ARCHITECTURES[a] for a in space.spec['base_architecture']['options']]}")
    if app.excluded_options:
        print(f"  dropped (tf_model_zoo absent): "
              f"{[BASE_ARCHITECTURES[a] for a in app.excluded_options['base_architecture']]}")

    categorical = ('keyframe_extraction', 'num_segments',
                   'base_architecture', 'consensus_function')

    for _ in range(200):
        design = space.sample(rng)
        check(space.validate(design), f"sampled design invalid: {design}")

        encoded = space.encode(design)
        check(
            encoded.shape == (space.encoding_length,),
            f"encoding length {encoded.shape} != {space.encoding_length}",
        )

        decoded = space.decode(encoded)
        check(space.validate(decoded), f"decoded design invalid: {decoded}")
        for gene in categorical:
            check(
                decoded[gene] == design[gene],
                f"round trip changed {gene}: {design[gene]} -> {decoded[gene]}",
            )
        for gene in ('learning_rate', 'dropout', 'gradient_norm_clipping'):
            check(
                abs(decoded[gene] - design[gene]) < 1e-4,
                f"round trip moved {gene}: {design[gene]} -> {decoded[gene]}",
            )

    # The meta-learner's design prediction is unconstrained regression, so
    # arbitrary vectors must still decode to something buildable.
    for _ in range(200):
        noise = rng.normal(size=space.encoding_length).astype(np.float32)
        decoded = space.decode(noise)
        check(space.validate(decoded), f"noise decoded to an invalid design: {decoded}")

    design = app.get_default_design()
    chromosome = design_to_chromosome(design)
    check(
        len(chromosome) == 7,
        f"the video chromosome must have 7 genes, got {len(chromosome)}",
    )
    check(
        chromosome_to_design(chromosome) == design,
        "design -> chromosome -> design is not the identity",
    )

    print("  design space: OK")
    return space


def verify_dataset(app):
    """The generated dataset and its loader must agree with each other."""
    print("\n[2/7] Generated dataset")

    info = app.dataset_info
    print(f"  {info['num_train_videos']} train / {info['num_val_videos']} val "
          f"videos, {info['num_classes']} classes, "
          f"{info['frames_per_video']} frames at {info['frame_size']}px")

    check(info['num_train_videos'] > 0, "no training videos were generated")
    check(info['num_val_videos'] > 0, "no validation videos were generated")
    check(
        info['frame_size'] >= 224,
        f"frames are {info['frame_size']}px, below the 224px the base models "
        f"need; the TSN augmentation crops rather than upscales",
    )

    # Every frame a list file claims must exist.
    for list_file in (app.train_list, app.val_list):
        for line in open(list_file):
            if not line.strip():
                continue
            directory, num_frames, label = line.strip().split(' ')
            frames = sorted(Path(directory).glob('*.jpg'))
            check(
                len(frames) == int(num_frames),
                f"{directory} lists {num_frames} frames but holds {len(frames)}",
            )
            check(
                0 <= int(label) < info['num_classes'],
                f"{directory} has out-of-range label {label}",
            )

    design = app.get_default_design()
    app._apply_design(design)

    inputs, targets = next(iter(app.train_loader))
    expected_channels = 3 * design['num_segments']
    check(
        inputs.shape[1] == expected_channels,
        f"a batch has {inputs.shape[1]} channels, expected "
        f"{expected_channels} (3 per RGB segment x "
        f"{design['num_segments']} segments)",
    )
    check(
        inputs.shape[-1] == app.model.crop_size,
        f"a batch is {inputs.shape[-1]}px, expected the model's crop size "
        f"{app.model.crop_size}",
    )
    check(
        targets.dim() == 1 and targets.shape[0] == inputs.shape[0],
        f"labels {tuple(targets.shape)} do not match the batch "
        f"{tuple(inputs.shape)}",
    )

    print(f"  batch: {tuple(inputs.shape)}, labels {tuple(targets.shape)}")
    print("  dataset: OK")


def verify_keyframes():
    """All five strategies must return the right number of in-range indices."""
    print("\n[3/7] Keyframe extraction")

    rng = np.random.default_rng(0)

    for value, name in sorted(KEYFRAME_STRATEGIES.items()):
        for num_frames, num_segments in [
            (16, 3), (16, 10), (100, 7), (3, 10), (1, 5), (5, 5),
        ]:
            indices = sample_indices(
                num_frames=num_frames,
                num_segments=num_segments,
                new_length=1,
                strategy=value,
                rng=rng,
            )
            check(
                len(indices) == num_segments,
                f"strategy {value} ({name}) returned {len(indices)} indices "
                f"for {num_segments} segments",
            )
            check(
                indices.min() >= 1,
                f"strategy {value} ({name}) returned a non-1-based index "
                f"{indices.min()}; TSNDataSet.get expects 1-based frames",
            )
            check(
                indices.max() <= num_frames,
                f"strategy {value} ({name}) returned index {indices.max()} "
                f"for a {num_frames}-frame video",
            )
        print(f"  strategy {value} ({name}): OK")

    print("  keyframe extraction: OK")


def verify_model_building(app, skip_architectures):
    """Every architecture and consensus function must build, run and backprop."""
    print("\n[4/7] Model building")

    num_classes = app.dataset_info['num_classes']

    print("  consensus functions:")
    for value, name in sorted(CONSENSUS_FUNCTIONS.items()):
        model = TSN(
            num_classes, 3, 'RGB', base_model='resnet18',
            consensus_type=name, dropout=0.5, partial_bn=False,
        )
        output = model(torch.randn(2 * 3, 3, 224, 224))
        output.float().sum().backward()

        gradients = [p for p in model.parameters() if p.grad is not None]
        check(
            gradients,
            f"consensus '{name}' produced no gradients; the recovered "
            f"implementation hand-wrote its backward pass",
        )
        # 'identity' deliberately keeps the segment axis; the others reduce it.
        expected_dim = 3 if name == 'identity' else 2
        check(
            output.dim() == expected_dim,
            f"consensus '{name}' returned a {output.dim()}-D output, "
            f"expected {expected_dim}-D",
        )
        # The application must turn either shape into a scalar loss.
        loss, logits = app._loss_and_logits(
            output.detach(), torch.zeros(2, dtype=torch.long)
        )
        check(loss.dim() == 0, f"consensus '{name}' did not give a scalar loss")
        check(
            logits.shape == (2, num_classes),
            f"consensus '{name}' gave logits {tuple(logits.shape)}, "
            f"expected (2, {num_classes})",
        )
        # Policy assignment must not reject the learnable consensus.
        model.get_optim_policies()
        print(f"    {value} ({name}): output {tuple(output.shape)}, OK")

    if skip_architectures:
        print("  base architectures: skipped (--skip-architectures)")
    else:
        print("  base architectures:")
        for value in sorted(BASE_ARCHITECTURES):
            name = BASE_ARCHITECTURES[value]
            if value in MODEL_ZOO_ARCHITECTURES and not model_zoo_available():
                print(f"    {value} ({name}): skipped (tf_model_zoo absent)")
                continue

            model = TSN(
                num_classes, 2, 'RGB', base_model=name,
                consensus_type='avg', dropout=0.5, partial_bn=False,
            )
            output = model(torch.randn(1 * 2, 3, model.crop_size, model.crop_size))
            check(
                output.shape == (1, num_classes),
                f"architecture '{name}' gave {tuple(output.shape)}, "
                f"expected (1, {num_classes})",
            )
            parameters = sum(p.numel() for p in model.parameters())
            print(f"    {value} ({name}): {parameters / 1e6:.1f}M params, OK")

    print("  model building: OK")


def verify_persistent_state(app):
    """Training must accumulate, and a design change must transfer weights."""
    print("\n[5/7] Persistent training state")

    design = app.get_default_design()
    app.reset_run_state()

    losses = []
    for timestep in range(4):
        metrics = app.exec_design(design, timestep)
        losses.append(metrics['train_loss'])

    print(f"  train loss over 4 timesteps: {[round(l, 4) for l in losses]}")
    check(
        all(np.isfinite(l) for l in losses),
        f"a timestep produced a non-finite loss: {losses}",
    )
    check(
        losses[-1] < losses[0],
        f"training loss did not fall over four timesteps "
        f"({losses[0]:.4f} -> {losses[-1]:.4f}); the network is probably "
        f"being rebuilt each timestep rather than carried forward",
    )

    # Change only the consensus function: the base model is untouched, so
    # nearly every parameter should survive.
    changed = dict(design)
    changed['consensus_function'] = 2 if design['consensus_function'] != 2 else 1
    metrics = app.exec_design(changed, 4)

    check(
        metrics['weights_transferred'] > 0.9,
        f"changing only the consensus function transferred "
        f"{metrics['weights_transferred']:.3f} of the weights; a change that "
        f"leaves the base model alone should keep nearly all of them",
    )
    print(f"  consensus change transferred "
          f"{metrics['weights_transferred']:.3f} of weights")

    # Change the base architecture: little should transfer.
    options = app.get_structured_design_space().spec['base_architecture']['options']
    other = next((a for a in options if a != design['base_architecture']), None)
    if other is not None:
        changed = dict(design)
        changed['base_architecture'] = other
        metrics = app.exec_design(changed, 5)
        print(f"  architecture change ({BASE_ARCHITECTURES[design['base_architecture']]}"
              f" -> {BASE_ARCHITECTURES[other]}) transferred "
              f"{metrics['weights_transferred']:.3f} of weights")
        check(
            app.num_design_changes >= 2,
            f"the application recorded {app.num_design_changes} design "
            f"changes, expected at least 2",
        )

    print("  persistent state: OK")


def verify_meta_features(app):
    """The meta-feature vector must be the same length at every timestep."""
    print("\n[6/7] Meta-features")

    design = app.get_default_design()
    app.reset_run_state()

    names = app.get_meta_feature_names()
    lengths = set()

    for timestep in range(3):
        features = app.extract_meta_features(timestep=timestep)
        lengths.add(len(features))
        check(
            list(features.keys()) == names,
            f"meta-feature keys at timestep {timestep} do not match the "
            f"declared schema",
        )
        check(
            all(np.isfinite(v) for v in features.values()),
            f"meta-features at timestep {timestep} contain a non-finite "
            f"value: { {k: v for k, v in features.items() if not np.isfinite(v)} }",
        )
        app.exec_design(design, timestep)

    check(
        len(lengths) == 1,
        f"meta-feature vector length varied across timesteps: {lengths}",
    )

    print(f"  {len(names)} features, identical length at every timestep: OK")


def verify_approaches(timesteps, videos_per_class, batch_size):
    """All four variants must run end to end and report a test performance."""
    print("\n[7/7] All four approaches")

    ga_params = {'population_size': 3}
    summary = {}

    for key, kwargs, runner in [
        ('onmar-accuracy', {'theta_t': None, 'theta_p': 0.85}, 'run_onmar'),
        ('onmar-design', {'theta_t': None, 'theta_p': 0.85}, 'run_onmar'),
        ('offmar-accuracy', {'theta_p': 0.85}, 'run_full_offmar'),
        ('offmar-design', {'theta_p': 0.85}, 'run_full_offmar'),
    ]:
        module = load_approach(key)
        approach_class = next(
            getattr(module, name) for name in dir(module)
            if name.startswith(('OnMAR', 'OffMAR'))
            and name not in ('OnMARConfig', 'OffMARConfig')
        )

        print(f"\n  --- {key} ---")
        start = time.time()
        approach = approach_class(
            application=make_app(videos_per_class, batch_size),
            meta_learner_type='knn',
        # k=2, not 3. The OffMAR variants split the timestep budget into two
        # phases, so Phase 1 contributes only timesteps/2 entries to the
        # knowledge repository, and scikit-learn's k-NN raises outright when
        # k exceeds the number of samples it was fitted on
        # ("Expected n_neighbors <= n_samples_fit"). k=2 keeps this script
        # runnable at the small budgets it is meant for.
            meta_learner_params={'k': 2},
            design_algorithm_params=ga_params,
            **kwargs,
        )
        results = getattr(approach, runner)(
            dataset_name='synthetic', timesteps=timesteps
        )

        test_performance = results.get('test_performance')
        check(
            test_performance is not None,
            f"{key} reported no test performance",
        )
        summary[key] = (test_performance, time.time() - start)
        print(f"  {key}: test={test_performance:.4f} ({summary[key][1]:.0f}s)")

    print("\n  summary:")
    for key, (performance, elapsed) in summary.items():
        print(f"    {key:18s} test={performance:.4f}  {elapsed:.0f}s")

    # A short run on an easy generated task should beat chance at least once;
    # requiring it of every variant would make this check flaky rather than
    # informative.
    chance = 1.0 / 4
    best = max(performance for performance, _ in summary.values())
    check(
        best > chance,
        f"no approach beat chance ({chance:.2f}); the best was {best:.4f}",
    )

    print("  all four approaches: OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--videos-per-class', type=int, default=8,
        help='Generated videos per class (small keeps this quick)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4,
        help='Videos per batch'
    )
    parser.add_argument(
        '--timesteps', type=int, default=4,
        help='Timesteps per approach run'
    )
    parser.add_argument('--skip-approaches', action='store_true',
                        help='Skip the four end-to-end approach runs')
    parser.add_argument('--skip-architectures', action='store_true',
                        help='Skip building every base architecture, which '
                             'downloads several hundred MB of weights')
    args = parser.parse_args()

    print("=" * 78)
    print("VIDEO CLASSIFICATION PIPELINE VERIFICATION")
    print("=" * 78)

    try:
        app = make_app(args.videos_per_class, args.batch_size)
        app.load_data()

        verify_design_space(app)
        verify_dataset(app)
        verify_keyframes()
        verify_model_building(app, args.skip_architectures)
        verify_persistent_state(app)
        verify_meta_features(app)
        if args.skip_approaches:
            print("\n[7/7] Approaches: skipped (--skip-approaches)")
        else:
            verify_approaches(args.timesteps, args.videos_per_class, args.batch_size)
    except CheckFailed as error:
        print(f"\n{'=' * 78}")
        print(f"VERIFICATION FAILED: {error}")
        print("=" * 78)
        return 1

    print("\n" + "=" * 78)
    print("ALL CHECKS PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
