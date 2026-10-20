# Faster simulation iteration and a 27-day geometry WAM study

This is a proposed engineering and research plan, based on the [simulation source audit](libero_simulation_evaluation.md). No rollout speedups or policy improvements have been measured in this PR. The local hardware inspection found an RTX 4080 with 16,376 MiB VRAM, an i9-13900KF with 24 physical cores/32 logical CPUs, and approximately 62 GiB RAM. These findings describe the current PC, not a remote training cluster.

## The recommended strategy

Use native MuJoCo/robosuite/LIBERO for paper scores. Start from the current FastWAM implementation, which already keeps a policy worker alive across tasks and supports compiled action inference. Profile one representative serial evaluation before building a batched server. Run a fixed small development panel on every promising checkpoint, a broader panel only on finalists, and full LIBERO plus LIBERO-Plus for the final comparison. [FastWAM manager][manager] [FastWAM inference][inference]

The fastest path to a useful idea is to reduce the number of expensive decisions: cache fixed training inputs and labels, eliminate repeated initialization, make failures reproducible, and reject unpromising model variants with a predefined development experiment. A faster physics engine cannot compensate for repeatedly tuning on a full 10,030-variant test set.

### Memory is the first gate on this PC

Fast-WAM's paper describes a 5B video backbone plus a 1B action expert, in addition to reused text/visual components. Six billion parameters at two bytes each already occupy roughly 12 GB in decimal units before activations, text encoder, VAE, temporary buffers and compilation caches. This arithmetic is a capacity warning, not a measured peak-memory estimate. Full-model Adam training also requires gradients and optimizer states, making an ordinary fully resident training run unsuitable for this 16 GB GPU. [Fast-WAM paper][fastpaper]

Before committing the month:

1. Load a released checkpoint with its matching normalization statistics and run one full policy query, recording actual GPU memory. Start eager to separate memory/compatibility issues from compilation.
2. Test whether precomputed language context allows the text encoder to remain off the GPU. The model exposes context/context_mask inputs, but the released evaluation path supplies prompts; exploiting the alternate interface needs adapter work.
3. Measure compiled inference memory separately, including cold compilation and warm queries.
4. Only then choose one policy worker and try 1, 2, then 4 simulator processes. Do not launch multiple copies of the 6B model on this GPU.
5. For learning experiments, prefer a small trainable geometry/action adapter, cached frozen features, or a smaller controlled model. A frozen backbone still consumes memory when resident; “freeze it” alone does not solve capacity. If a representative training batch cannot run by day 3, reduce scope or obtain larger training hardware rather than spending the remaining month on offload machinery.

Quantization, fewer denoising steps and changed precision can be useful model variants, but they need their own success-rate validation. No released-checkpoint inference fit or training throughput has been established here. The inspected system Python lacks torch, mujoco, robosuite, libero and hydra; other Python environments were not exhaustively surveyed.

## Measure the actual bottleneck

Break an episode into disjoint timings:

~~~text
T_episode = T_reset_and_restore + T_settle
          + sum(T_policy_query)
          + sum(T_control_and_physics + T_render_and_observation + T_corruption)
          + T_transport + T_logging_and_video
~~~

Count policy queries and executed control steps alongside seconds. Record simulator construction, model loading and compilation separately from warm evaluation. On GPU, synchronize at timing boundaries or use properly synchronized CUDA events; host dispatch duration alone undercounts inference. Avoid adding synchronization to every small operation in the final throughput run.

Measure four workloads on the same pinned machine:

| Workload | Purpose | Preserve |
| --- | --- | --- |
| Fixed-action, no policy, no camera | CPU mechanics/controller ceiling | Every physics/control step and predicate check |
| Same actions with required cameras | Incremental rendering/observation cost | Camera setup, resolution and preprocessing |
| Frozen observation corpus, policy only | Query latency and memory | Denoising, precision and real input shapes |
| Closed-loop baseline | End-to-end cost and success | Entire evaluation protocol |

Add explicit corruption timing for Plus. Its glass blur contains per-pixel Python loops, and its motion blur encodes/decodes images through an external image library. A policy-free camera benchmark can therefore miss a large Plus-specific cost. [Plus corruption implementation][plusnoise]

For each configuration, run warmups and a fixed representative panel, report median and p95 episode time, episodes/hour, model queries/second, maximum VRAM, CPU use, errors and successes. Publish raw per-episode records. Report a fixed-action speed comparison alongside closed-loop speed: a better policy that finishes sooner changes the workload and can otherwise masquerade as a faster simulator.

**Planning example, not a measurement:** at 20 seconds/episode, 2,000 episodes require 11.1 serial hours and 10,030 require 55.7 hours. At 60 seconds/episode, they require 33.3 and 167.2 hours. At four effectively independent workers, the idealized 55.7-hour workload becomes 13.9 hours, but a shared GPU model and renderer prevent assuming a fourfold gain. Estimate the real budget from measured completed episodes/hour.

Amdahl's law is useful: if rendering occupies fraction f of runtime and becomes k times faster, the total speedup is 1 / ((1-f) + f/k). For f=0.6 and k=10, the bound is about 2.17×, not 10×. Use this to prioritize work rather than promising an unmeasured speedup.

## Optimization priorities

| Priority | Change | Why it may help | Validation required |
| --- | --- | --- | --- |
| 1 | Reuse model workers and compiled kernels | Avoid repeated model load/compile | Already supported; measure warmup separately. |
| 1 | Disable or sample rollout video | Avoid image copies, MP4 encoding and I/O | Preserve policy images; compare action traces. |
| 1 | Cache language embeddings by exact instruction | Avoid repeated T5 calls; potentially free encoder VRAM | Include tokenizer/model/precision in cache key; Plus language variants must not collide. |
| 1 | Verify EGL GPU rendering | Avoid accidental software rendering | Check actual backend and device, not just an environment variable. |
| 2 | Render only when policy needs new observations | Reduce camera work within open-loop action chunks | Preserve physics, predicates, observation semantics and corruption RNG stream. |
| 2 | One model server, a small simulator process pool | Overlap CPU work and batch model queries | Per-episode ordering, batching equivalence and memory tests. |
| 3 | Fast reset or a complete settled-state cache | Amortize model/renderer/reset/settling work | Full state equivalence across task types and histories. |
| Later | GPU physics port | Potential large-batch gains | New backend and renderer validation; substantial engineering. |

### What the current FastWAM code already does, and what it does not

The manager supplies persistent workers, dynamic task assignment and recovery. The inference path supports compiled prefill/action denoising and cached visual keys/values within a query. Reimplementing these features is not a sensible first milestone. The current evaluator, however, always calls save_rollout_video after each trial and stores per-step replay images. Setting visualize_future_video=false only disables imagined-video output, not rollout MP4s. A true no-video option would need to guard both capture and encoding. [manager][eval][inference]

The configuration lists env_num, and get_libero_env can build SubprocVectorEnv when passed env_num > 1. The current _evaluate_task call does not pass that field, and run_single_episode expects one environment. Therefore EVALUATION.env_num=4 is not a working four-environment speedup for this evaluator. Either keep the existing serial worker or implement and validate a real pool; document new flags as new functionality. [eval][utils]

The repeated prompt path in _predict_action_chunk reaches encode_prompt in infer_action. Cache the exact language context once per task/variant, then use the model's existing context input. This is a concrete candidate optimization, not a feature claimed to exist in the evaluator today. [eval][inference]

### Render less without changing what the policy sees

For FastWAM's default 10-action execution prefix, the policy consumes a new observation at the next query, while intermediate camera frames mainly support videos and observation updates. A lower-level adapter could suppress unused camera observations during those intermediate steps and still execute all controller updates, physics steps, task-state updates and success checks. The theoretical reduction is in render calls, not in physics work or guaranteed total runtime.

Do not replace ten env.step calls with one larger physics step or skip intermediate success checks. Do not simply reuse a stale image at the next query. Validate image equality at every query boundary against the reference path.

Plus is harder: skipped corruptions can consume fewer random numbers and change later observations, even if skipped images were never fed to the policy. The reference-equivalence lane must preserve RNG advancement, or precompute equivalent corruption parameters indexed by control step. Redesigning the RNG by frame index is cleaner for a new harness but defines a new protocol unless shown to match the original. For complex stochastic corruptions, retain the reference path until this is solved. [plusnoise]

Removing video output also does not automatically remove rendering, because rendering is performed by camera observables inside the environment. These are two different optimizations.

### Parallelism on one GPU

Use CPU processes for independent MuJoCo environments, one renderer context owned by each worker as supported by the backend, and one GPU policy owner. Avoid forking after creating CUDA/OpenGL contexts; initialize them in the worker process. With a single shared GPU, rendering and model execution compete for memory and compute, so benchmark 1/2/4 workers before considering 8.

Each worker sends an episode ID, observation, exact instruction/context key, proprioception and replan index. The model server batches compatible requests and returns action chunks tagged with those IDs. Each episode waits for its own result; batching must not introduce dropped observations or stale actions. The server should use stable per-episode random inputs so queue order does not change action samples. Partial completion must not let inactive environments contaminate averages.

Separating model and simulator Python environments can avoid the old LIBERO/Plus dependency stack conflicting with FastWAM's current dependencies. OpenWAM provides an existing policy-server/simulator-client design with separate environments, so inspect it as an integration reference before inventing a transport. It is not evidence that FastWAM's model can be dropped into that server unchanged. [OpenWAM evaluation guide][openwam]

Prefer local lossless arrays/shared memory over JPEG compression: changing image compression changes policy observations. Profile transport before optimizing it; the model may dominate total time.

### Reset acceleration has a correctness cost

Hard resets can rebuild simulation and renderer state. Soft resets save work, but LeRobot explicitly warns that they are not bit-identical after settling and recommends hard resets for benchmark reproduction. Switching hard_reset=false is therefore an experimental alternative, not a certified transparent speedup. [LeRobot reset guidance][lerobot]

A complete reusable snapshot must account for MuJoCo integration state, model-side changes, controller goals/history, observation caches and RNG state, not just qpos/qvel. Cache it after the prescribed settling sequence and key it by the full protocol and episode identity. Warm-start accelerations can affect exact replay. Validate snapshots on contact-rich and articulated tasks, across repeated resets and after different preceding episodes. [MuJoCo state documentation][mjsim]

For Plus, variants can change model geometry, cameras, robot identity and textures. Reuse only within equivalent model/asset configurations. Since the official protocol uses one episode per variant, per-variant reset caching may have a poor first-run payoff; it becomes more valuable across many checkpoints, at the cost of storage and cache validation.

### Why not move straight to MJX, Warp or another simulator

Accelerated MuJoCo backends are real options, particularly for large batches. Current MJWarp supports rendering and many MuJoCo features, so it would be inaccurate to dismiss it as physics-only. However, its documented precision and feature differences, contact buffer capacities and renderer differences require validation. Moving LIBERO predicates, controller behavior, assets and images onto a new backend is a project of its own. [MJWarp documentation][mjwarp]

For this deadline and one 16 GB GPU, the recommended default is CPU MuJoCo physics plus GPU rendering and inference. A port is justified only if profiling shows physics dominates after cheaper improvements, and a validated existing port covers the required tasks. Final reference scores should still use the pinned native evaluator unless the community protocol explicitly accepts the alternative. A learned GeoWM rollout is also not a substitute for native-simulator success evaluation; its prediction error would become part of the judge.

## Reduce iteration cost without weakening final evidence

Define development manifests before inspecting new-method results. Keep a final holdout manifest unused for choosing architecture and hyperparameters. LIBERO-Plus is a test benchmark; repeatedly selecting models on its full results contaminates its role as final evaluation. If official variants are used in a development panel, disclose that and reserve disjoint variants/base-task groups as feasible; do not claim the full set was untouched.

| Stage | Proposed workload | Decision |
| --- | --- | --- |
| Smoke | 2 tasks × 2 starts; one placement and one articulated task | Action signs, input alignment, no crashes, plausible baseline behavior |
| Daily development | 8 tasks × 5 starts = 40 clean episodes, plus about 140 stratified perturbation variants | Reject broken or clearly weak ideas; do not interpret small changes as significant |
| Candidate validation | 40 tasks × 10 starts = 400 clean episodes, plus about 700 stratified variants | Compare finalists and check category tradeoffs |
| Final | 2,000 clean episodes plus all 10,030 Plus variants, one rollout each | Main paper tables on a frozen protocol |

These smaller panels are proposed development experiments, not official benchmark scores. Cover task type, base task, perturbation category and available difficulty labels, including unknown labels. Stratification is important because easy successes and one class of failure can otherwise dominate a tiny sample.

Use matched episode keys and random inputs for baseline and method. Analyze paired success differences; confidence intervals that cluster by underlying task address correlated variants better than treating all 10,030 as independent tasks. Also report per-training-seed variation if multiple trained checkpoints are available. A simulator seed repetition is not a substitute for a training seed. Small panels are triage instruments, not enough evidence for a one-point improvement near saturation.

Freeze an infrastructure retry policy. Re-run crashed infrastructure entries with the same inputs, never replace policy failures with new seeds until success. Resuming should skip only validated completed entries with matching protocol/checkpoint fingerprints. Failed setup must not shrink the denominator.

## A focused GeoWM hypothesis

**Proposed claim to test:** explicit future-geometry supervision improves camera/layout robustness of an action policy without requiring full future RGB generation at inference.

This uses GeoWM's explicit geometry and direct horizon prediction, but adds language, robot state and action learning. The source paper does not establish action-conditioned counterfactual forecasting, so that extension must be tested. A model trained only on observational trajectories must not be assumed to predict the consequences of arbitrary alternative actions. [GeoWM][geowm]

Start with training-only future-depth supervision on shared observed features. Keep the action branch causally separated from ground-truth future tokens. Then compare a compact predicted-geometry input at inference if the first result is promising. For an action-conditioned forecaster, use the demonstrated action prefix during training; for planning, evaluate candidate actions without accessing future state. Do not make action prediction depend circularly on a future that already requires the unknown action.

Use these controls:

1. Same action model and data, without geometry supervision.
2. Current-depth/geometry supervision only, to isolate the benefit of prediction.
3. Future geometry at one or several direct horizons.
4. Matched-capacity RGB/feature auxiliary supervision, to test whether gains are geometric rather than just extra parameters or loss.
5. Predicted geometry versus simulator-depth oracle, clearly separated by input privileges.
6. If using inference-time geometry, remove it at inference and measure the change; include geometry extraction/forecasting in total latency.

Track clean success, camera/robot/layout and remaining category success, end-to-end policy latency, peak VRAM and training cost. Evaluate depth error on task-relevant moving regions as a diagnostic, since whole-image error can be dominated by a static table. Avoid claiming that better depth explains better control without an ablation connecting them.

A necessary related-work comparison is **Faster-WAM**, distinct from Fast-WAM. Its August 2026 paper argues that inference-time future conditioning improves robustness and reports 73.57% versus its Fast-WAM baseline's 49.14% on LIBERO-Plus. These are that paper's results, not a universal current FastWAM score. Efficient future conditioning itself is therefore already occupied territory; the new contribution needs to isolate explicit geometry, action relevance and robustness under a matched protocol. [Faster-WAM][fasterpaper]

For the month-long scope, prefer one model family and one clear mechanism. Demonstrating a reliable effect on a smaller controlled model plus a targeted larger-model experiment is more defensible than an incomplete large-model reproduction.

## Calendar and concrete gates

From 20 October 2026, CVPR 2027's listed registration deadline is 10 November and paper deadline is 16 November, both Anywhere on Earth. The paper deadline is 27 calendar days away, not a full 30-day development window. [Official dates][cvpr]

| Dates | Deliverable and stopping rule |
| --- | --- |
| Oct 20–22 | Lock evaluator, run checkpoint memory/compatibility smoke test, establish replay and timing baseline. If the baseline cannot run, choose a smaller controlled model immediately. |
| Oct 23–25 | Make only the highest-value evaluated speed fixes; reconstruct/capture geometry labels and verify RGB-depth/action alignment. Freeze development and holdout manifests. |
| Oct 26–30 | Train baseline, current-geometry control and future-geometry variant. Require an effect beyond development noise before expanding experiments. |
| Oct 31–Nov 4 | Run matched ablations and camera/layout analysis. If geometry does not help control, revise the claim or stop the proposed direction; do not select isolated favorable tasks. |
| Nov 5–9 | Freeze final recipe and begin complete reference runs with room for infrastructure retries. Draft method, protocol and limitations. |
| Nov 10 | Registration milestone; audit remaining experimental coverage and uncertainty. |
| Nov 11–14 | Complete tables, failure analysis and reproducibility artifacts. Avoid new architectures. |
| Nov 15–16 | Final writing and submission buffer. |

The first implementation milestone should be a reliable, measured baseline and a cheap fixed development loop. That creates the capacity to evaluate the geometry idea repeatedly; a new simulator backend does not fit this schedule unless it is already validated.

## Reproducing the source audit

Run from the repository root with standard-library Python:

~~~bash
python3 scripts/research/audit_libero_plus_manifest.py   --output docs/research/libero_plus_manifest_audit.json
~~~

The command downloads two files from an immutable revision, parses the Python task map with ast.literal_eval, verifies every name/index correspondence, computes category/difficulty counts and records SHA-256 hashes. It does not execute downloaded Python or import the simulator. The --source-dir option instead audits the same paths in a local checkout and deliberately removes the claimed upstream revision from its output.

The checked-in JSON is a measured source-data result. All worker-count choices, cache designs, runtime examples, timeline gates and geometry improvements above are proposals awaiting runtime validation.

[manager]: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/libero/run_libero_manager.py
[eval]: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/libero/eval_libero_single.py
[utils]: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/libero/libero_utils.py
[inference]: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py
[plusnoise]: https://github.com/sylvestf/LIBERO-plus/blob/4976dc30028e805ff8094b55501d532c48fec182/libero/libero/envs/env_wrapper.py
[openwam]: https://github.com/OpenWAM-Official/OpenWAM/blob/main/benchmarks/libero/README.md
[lerobot]: https://huggingface.co/docs/lerobot/main/en/libero
[mjsim]: https://mujoco.readthedocs.io/en/stable/programming/simulation.html
[mjwarp]: https://mujoco.readthedocs.io/en/latest/mjwarp/
[geowm]: https://arxiv.org/html/2610.07381v1
[fastpaper]: https://arxiv.org/html/2603.16666v1
[fasterpaper]: https://arxiv.org/html/2608.04404v1
[cvpr]: https://cvpr.thecvf.com/Conferences/2027/Dates
