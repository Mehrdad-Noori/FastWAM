# LIBERO simulation and evaluation for geometry world action models

Research snapshot: 20 October 2026. FastWAM source revision: 7faa71108368fbb3b6885649f112af607427a2d4. This report distinguishes the released simulator, a particular model's evaluator, and proposed experiments. It is a source audit, not a measured simulation speed benchmark.

**Recommendation:** keep native LIBERO as the reference evaluator, freeze its complete protocol, and use a small fixed development panel before running full LIBERO and LIBERO-Plus. For a GeoWM follow-up, test whether explicit future geometry improves action success under camera and layout changes. Simulation acceleration should support that experiment rather than become a second research project.

Read next: [speed and iteration strategy](simulation_iteration_plan.md), [source audit output](libero_plus_manifest_audit.json), and the [reproducible audit script](../../scripts/research/audit_libero_plus_manifest.py).

## What the simulator does

A robot policy receives images, language, and whatever proprioception its interface permits. It returns commands. LIBERO executes those commands in a simulated scene and returns the next observation. The model's imagined future is not used as the environment or as the success judge. Even a WAM that predicts beautiful future video fails the benchmark if its executed actions do not satisfy the task's simulator predicates.

The implementation stack is:

~~~mermaid
flowchart LR
  T[Task BDDL and assets] --> E[LIBERO environment]
  I[Released initial state] --> E
  E --> O[RGB cameras and proprioception]
  O --> P[Policy or WAM]
  L[Language instruction] --> P
  P --> A[Normalized action chunk]
  A --> D[Denormalize and map gripper]
  D --> C[robosuite operational space controller]
  C --> M[MuJoCo physics]
  M --> E
  M --> J[Task predicates on simulator state]
  J --> S[Success or continue until timeout]
~~~

LIBERO provides tasks, assets, initial states and predicates. Robosuite builds robot environments, computes control torques, handles observations and rendering. MuJoCo integrates the mechanics and contacts. An evaluator outside that stack selects episodes, invokes the policy, decides when to stop, and aggregates results. These are separate places where experimental settings can change. [L1][L2][R1]

### Actions for someone coming from geometry forecasting

The default wrapper uses a Panda arm and OSC_POSE controller. Its usual action has seven components: three end-effector position increments, three rotation increments represented in axis-angle coordinates, and one gripper command. These are neither seven joint angles nor seven motor torques. The controller converts a desired end-effector motion into joint torques while the physical simulator determines what actually moves. Contact, joint constraints and controller tracking mean that adding a requested displacement does not guarantee that exact displacement in the next state. [L1][R2]

In robosuite 1.4.0's default OSC_POSE configuration, input components in [-1, 1] map to position increments up to ±0.05 m and rotation-vector components up to ±0.5 radians. The resolved controller config is authoritative; these are not universal units for every VLA checkpoint. The controller uses its current end-effector position and orientation to construct its target. A camera-frame displacement must be transformed into the controller's frame before use; under camera perturbations the image axes and action axes do not rotate together. [R2][R3]

FastWAM first reverses its dataset normalization, maps gripper openness from [0, 1] to [-1, 1], reverses its sign, and optionally binarizes it. Its environment convention is -1 open and +1 close. A sign error can look like a failed grasping method. Preserve the checkpoint's statistics file, camera ordering and action convention. Its image helper rotates both images 180 degrees to match its preprocessing. [F1][F2]

Three clocks matter:

- Physics: robosuite 1.4.0 defaults to a 0.002 s integration step.
- Control: LIBERO's wrapper defaults to 20 Hz, or 0.05 simulated seconds per action. With those defaults, one action causes 25 physics steps and controller updates.
- Policy: FastWAM's current configuration executes 10 predicted actions before querying the model again. That is 0.5 simulated seconds between queries, although the predicted chunk can contain 32 actions. [R1][R4][L1][F1][F3]

Wall-clock time is a fourth, separate quantity. Synchronous evaluation normally waits for inference without advancing the world. A slower policy can therefore obtain the same success rate while taking much longer to evaluate. Lowering policy latency does not change the 20 Hz physics protocol. Increasing the number of actions executed before replanning does change the feedback policy.

### Observations and resets

The default LIBERO wrapper exposes an external agentview camera and a wrist camera named robot0_eye_in_hand. It supports depth and segmentation, but depth is off by default. Low-dimensional observations include robot state; the actual policy interface decides which fields it consumes. Do not expose object poses, goal predicates or segmentation labels to a nominal RGB policy merely because they are accessible to the evaluator. [L1]

A typical episode is:

1. Select the exact task and released initial-state index.
2. Create its environment, or reuse an environment with the same task definition.
3. Seed it and reset it. A hard reset can rebuild the model, simulator and renderer.
4. Install the released state with set_init_state. This sets flattened simulator state, runs forward computation and refreshes observations.
5. Execute the evaluator's settling actions.
6. Obtain the policy's input, predict an action chunk, and execute its specified prefix one action at a time.
7. Check success after each environment action. Stop on success or the evaluator's action budget.
8. Record an outcome for that manifest entry and aggregate over the prescribed entries. [L1][R1][F1]

FastWAM creates one environment per task and resets it across trials. Its model worker persists across tasks. These are different lifetimes: persistent model loading does not eliminate the environment's hard-reset cost. [F1][F4]

## How success is judged

**There is no human, image-similarity model, learned reward model or LLM in the ordinary LIBERO success check.** The task's BDDL goal is evaluated against privileged simulator state. Language is the policy input; the symbolic goal supplies the scoring rule. [L2][L3]

For example, the released microwave task asks for the yellow-and-white mug to be placed inside and the microwave closed. Its goal is a conjunction of In(mug, heating_region) and Close(microwave). Both must be true in the same checked state. The problem implementation loops over the parsed goal predicates and combines their Boolean results. It does not require the policy to narrate its plan. [L3][L4]

Predicate details matter:

| Predicate | What the code tests | Consequence |
| --- | --- | --- |
| In | Destination contact and containment checks | A site destination's contact check always returns true; in that case containment supplies the effective condition. |
| On | Destination check_ontop | For ordinary objects this includes contact, relative height and horizontal center distance below 3 cm. Sites use their own region logic. |
| Open / Close | Object-specific joint-state tests | The object's implementation supplies thresholds; there is no one opening angle shared by all fixtures. |
| TurnOn / TurnOff | Object-specific joint or switch checks | A rendered light is not the primary judge. |
| Conjunction | All goal predicates are true together | Completing separate parts at different times is insufficient if they do not coexist. |

These are implementation predicates, not a universal physical definition of containment or stable placement. In particular, site containment can check an object's representative position rather than containment of every mesh vertex. Inspect the task, state class and asset for a boundary case. [L3][L5][L6]

The base domain's step replaces the returned done value with _check_success(). This is why FastWAM can interpret done as task success for this specific environment. Generic Gym/Gymnasium wrappers can use termination or truncation differently; a timeout must never become a success merely because a generic done flag is true. [L2][F1]

Success is checked at control-step boundaries, not continuously at every physics substep. FastWAM stops at the first successful action. It does not impose an extra multi-second stability interval or a universal gripper-release condition. A diagnostic such as “success persists for another second” could be informative, but would be an additional metric, not a silent replacement for official success. Likewise, partial subgoal completion is useful for diagnosing failure but does not replace binary task success.

## Which evaluation protocol is being reproduced

LIBERO originally targets lifelong and multitask transfer. It contains Spatial, Object and Goal suites with ten tasks each, plus 100 other tasks split into LIBERO-90 and LIBERO-10. Modern VLA/WAM papers usually report the four ten-task suites, calling LIBERO-10 “Long.” Training and testing semantics differ between the original lifelong study and subsequent multitask imitation learning papers. “Evaluated on LIBERO” alone is insufficient to identify an experiment. [L7]

The original released standalone evaluator uses 20 vectorized environments, five zero-action settling steps and a configured 600-step cap. It accumulates whether each environment ever succeeded. Its default evaluation configuration also lists n_eval=20. The popular 50-trial recipe should not be presented as the only original LIBERO protocol. [L8]

### Current FastWAM versus OpenVLA-OFT

These values come from the pinned source revisions, not an assumed benchmark-wide default:

| Setting | FastWAM 7faa711 | OpenVLA-OFT e4287e9 |
| --- | --- | --- |
| Trials per task | 50 | 50 |
| Spatial action cap | 400 | 220 |
| Object action cap | 400 | 280 |
| Goal action cap | 400 | 300 |
| Long action cap | 700 | 520 |
| Settling steps before the action budget | 30 | 10 |
| Actions executed per policy query by default | 10 | 8 |

The four-suite full evaluation therefore contains 40 × 50 = 2,000 episodes, but the action opportunities differ substantially between these evaluators. FastWAM's current full-run upper bound is 950,000 policy-controlled actions plus 60,000 settling actions; early success reduces that number. These are calculated counts, not timings. FastWAM's paper states 2,000 trials but does not establish that every detail of the current release was used for the paper. [F1][F3][O1][P2]

Use two explicitly named experiments when necessary: reproduction of the released baseline recipe, and a matched-protocol comparison in which all compared policies use the same action caps, settling process, observations, action conversion and task states. Changing chunk execution length is a policy ablation. Do not quietly shorten horizons to obtain faster benchmark results.

For equal trials per task, average binary episode success within each task, then average tasks within a suite. The equally weighted mean of the four ten-task suites equals the overall episode mean when all entries are complete. Missing or crashed episodes break that equivalence if a results script silently drops them. Report completed/expected entries and keep infrastructure errors separate until the predefined rerun policy resolves them.

## What LIBERO-Plus changes

LIBERO-Plus modifies LIBERO across seven axes: object layout, camera viewpoint, robot initial state, language, illumination, background texture and sensor noise. It uses the same kind of symbolic task judge. Robustness concerns whether the controller succeeds under the modified inputs and scene configuration, rather than whether an image looks robust. The official README requests one trial per variant instead of 50. The benchmark's 10,030 variants are not 10,030 wholly independent semantic tasks. [P3][PLUS1]

| Dimension | Main change | Geometry relevance |
| --- | --- | --- |
| Camera | External viewpoint, orientation or field of view | Tests coordinate consistency, occlusion handling and calibration dependence. |
| Robot | Starting arm configuration | Tests reachability and proprioceptive grounding, beyond static scene reconstruction. |
| Layout | Target displacement or confounding objects | Tests spatial relations and distinguishing target from distractors. |
| Language | Instruction rewrites | Geometry alone cannot identify an arbitrary requested goal. |
| Light | Illumination changes | Can corrupt RGB-derived depth even with fixed physical geometry. |
| Background | Surface/scene appearance | Tests dependence on texture shortcuts. |
| Noise | Image corruptions | Tests the complete perception pipeline; geometric inference is not automatically immune. |

The geometry column gives proposed interpretations, not demonstrated GeoWM results.

### Release audit and counting pitfalls

At revision 4976dc30028e805ff8094b55501d532c48fec182, both the task-map file and classification JSON agree on the following counts. The included audit checks every name and its index, without importing LIBERO:

| Suite | Released map and JSON | arXiv v2 Table 7 |
| --- | ---: | ---: |
| Spatial | 2,402 | 2,293 |
| Object | 2,518 | 2,576 |
| Goal | 2,591 | 2,569 |
| Long | 2,519 | 2,592 |
| Total | 10,030 | 10,030 |

Across the released categories, counts are camera 1,599; robot 1,550; language 1,537; light 1,142; background 1,076; noise 1,601; layout 1,525. The total and category totals match the paper, while allocation within suites differs. This audit does not establish why. Use the pinned manifest for execution and report its counts. Do not silently repair the source to match the paper table. [PLUS2][P3]

The classification JSON uses IDs starting at 1. The benchmark API's task indices start at 0. At this release, every row's name matches task_map[id - 1]. Join results by suite and exact task name, or explicitly convert the verified ID convention. An unadjusted integer join mislabels results.

There are also **121 null difficulty labels**, all in Goal light variants. Keep those variants in the overall and light scores. For difficulty breakdowns, report an unknown bucket; do not discard them or invent levels. Difficulty levels are based on how many of four reference models succeeded, not simply camera angle or physical perturbation severity. [PLUS2][P3]

For the headline aggregate, report the micro-average over all 10,030 required variants, plus per-category and per-suite scores. Since category and suite counts differ, a mean of seven category percentages or four suite percentages is a different statistic. Label any macro-average separately. To reproduce a particular published table, verify that paper's aggregation; the README does not fully specify a scoring implementation. The micro-average here is the explicitly recommended, auditable convention.

A single rollout per variant measures performance on that fixed manifest. It does not measure uncertainty over policy noise or newly sampled starts. Additional repeated-seed results are useful, but identify them as additional experiments. Save every result so paired comparisons can reuse exactly the same variant and seed identities.

### FastWAM is not currently a drop-in LIBERO-Plus evaluator

Static inspection identifies several integration problems before any model is loaded:

1. **Path type:** FastWAM passes a pathlib.Path as bddl_file_name. The pinned Plus wrapper performs string membership and split operations on it. Convert the path to a string at the adapter boundary.
2. **Initial-state lookup:** FastWAM directly constructs and loads task.init_states_file. Plus's get_task_init_states resolves suffixes for camera, language, light, backgrounds and layouts, with special paths and shapes for layout variants. Use that benchmark API after validating the pinned implementation, rather than assuming each variant filename has a physical state file of the same name.
3. **Dependency conflict:** FastWAM pins NumPy 2.2.6; Plus pins an older NumPy and its fog code uses np.float_, removed in NumPy 2. Independently lock model and simulator environments, or explicitly validate a compatibility patch. Installing one package over the other is not a reproducibility strategy.
4. **Trials:** change the FastWAM evaluator's num_trials to 1 for the Plus release protocol, and enumerate actual benchmark n_tasks. Its manager already queries n_tasks, so a hard-coded ten-task launch is unnecessary.
5. **Noise path:** the Plus wrapper applies corruptions in reset and step, but its set_init_state regeneration returns observations directly. Preserve the settling process and verify which observation is first fed to the model.

These are code-level findings, not a claim that an integrated Plus rollout has been tested. FastWAM's pyproject also does not install LIBERO, robosuite or MuJoCo. Its dataset directory references MuJoCo 3.3.2, but a directory name is not an installed version lock. [F1][F2][F4][F5][PLUS3][PLUS4]

## Why repeated evaluations can differ

**MuJoCo is deterministic given the same complete integration state, model, controls and numerical execution conditions.** Its documentation limits exact reproducibility to the same version and architecture and explains that tiny differences can amplify during contact. Therefore “physics is random” is an inadequate diagnosis. [M1]

There are several distinct sources of variation:

| Layer | Specific mechanism | How to isolate it |
| --- | --- | --- |
| Episode selection | Different task orders, initial-state indices or variants | Compare exact manifest keys and hashes. |
| Reset | Placement and fixture sampling; randomization retries | Compare model fields and state immediately after reset and after state restoration. |
| Incomplete restoration | Serialized state omits integration/controller/observation history | Compare native full state plus controller internals before the first action. |
| Observation | Different camera setup, renderer, preprocessing or random corruption | Compare raw and preprocessed images for the same physical state. |
| Policy sampling | Action diffusion/flow noise, dropout or policy state | Repeat inference on a frozen input with explicit RNGs and eval mode. |
| Numerical execution | Different libraries, kernels, precision, devices or batches | Compare actions before investigating contact divergence. |
| Scheduling | Random streams coupled to order, batching or resume position | Run identical episodes alone, reordered and resumed. |

The LIBERO wrapper delegates seed to np.random.seed in the domain. This does not seed every Python or policy RNG. Plus generates nondefault task permutations with Python random.shuffle during module import. Use task order 0 and export explicit names, rather than assuming a seed set after import controls those permutations. [L1][L2][PLUS4]

The robosuite 1.4.0 flattened state stores time, qpos and qvel, and asserts no actuator activation state for that representation. It does not serialize all MuJoCo integration fields, qacc_warmstart, model-side fixture transforms, controller targets, observation caches or external RNGs. LIBERO's reset can modify fixture body_pos and body_quat in the model; installing qpos/qvel does not inherently overwrite those fields. This supplies concrete reasons to test fixed-state resets, but does not prove every task exhibits drift. [R5][L2]

In Plus, motion blur samples an angle; fog samples a random field; glass blur samples local pixel offsets. These use NumPy randomness while processing images. Thus two runs can agree in qpos and still receive different observations. Gaussian and zoom blur paths should not be lumped together with the random paths. Only the external camera is corrupted in the inspected wrapper. [PLUS3]

FastWAM's inference interface passes the configured seed to each prediction. Its action-only inference creates a new generator with that seed, so with a fixed seed and shape it can reuse the same initial noise each query. Do not assume a fresh unseeded draw explains every repeated-run difference. Conversely, changing to a continuously advancing generator is a protocol change. Its global seeding utility seeds Python, NumPy and Torch but does not itself enable all deterministic-kernel controls. [F1][F6][F7]

PyTorch documents that seeding alone does not guarantee identical results across releases or platforms. Deterministic algorithms, disabling cuDNN benchmarking and relevant CUDA settings can help isolate algorithmic nondeterminism. They can cost performance or expose unsupported operations. Record precision, compilation and batching along with seeds, and keep a diagnostic deterministic mode separate from an accelerated mode whose equivalence has been tested. [T1]

### A practical determinism experiment

Use one easy placement task, one articulated-object task and one long conjunction task, then add one camera variant and one stochastic-noise variant. Repeat each five times as an initial diagnostic. This is a proposed test plan, not a measured result.

1. **Fixed actions, physics only:** save a complete post-settling state and execute an identical recorded action sequence in fresh processes. Compare qpos, qvel, controller targets, integration inputs and predicate values at every control step. A replay beginning from a partial state is not a valid test of full-state determinism.
2. **Fixed states, rendering:** render identical physical states and compare raw image hashes and maximum pixel differences. Then compare policy preprocessing separately.
3. **Fixed inputs, policy:** replay an immutable observation/language/proprioception corpus through the model; compare denormalized actions. Test eager/compiled and each intended batch size independently.
4. **Closed loop:** combine the layers with a fixed manifest. Log the first action, image or state divergence, not just the final success bit.
5. **Scheduling:** run the same manifest serially, with candidate worker counts, in reversed order, and after interruption/resume. Episode identities and inputs must survive these transformations.

Use exact equality when testing a genuinely identical stack, plus maximum numerical differences and the earliest divergence step to diagnose a failure. A tolerant numerical match is not evidence of bitwise determinism. Even if final success matches, changed action traces can reveal protocol drift.

For a new evaluation harness, derive independent environment, corruption and policy seeds from a stable digest of base seed, suite, exact task name and episode index. Optionally key policy noise by replan index and image corruption by control step/camera. Do not use Python's process-randomized hash or worker rank as the episode identity. Preserve upstream seeding in a reproduction lane; adopting independent per-episode streams defines an explicitly documented new lane.

### What must be logged

At minimum, save source revisions, package locks, CUDA/driver versions, GPU/CPU, renderer backend, checkpoint and statistics hashes, task BDDL and assets identifiers, initial-state file/index/hash, resolved controller settings, camera calibration/resolution/preprocessing, timestep and control frequency, settling actions/count, caps, prediction and execution horizons, denoising settings, seed strategy and batching/compile/precision settings.

For each episode, store manifest key, success, terminal reason, first-success step, action count, reset and policy timing, total wall time, errors/retries and diagnostic artifact paths. A resume cache must be keyed by protocol and checkpoint fingerprints. FastWAM currently checks result-file existence by task; using a fresh output directory per protocol/checkpoint avoids accidentally accepting stale files. An infrastructure timeout should be recorded distinctly from a policy that exhausted its action budget. [F4]

## How GeoWM can use this simulator

GeoWM forecasts explicit future depth at requested horizons from geometric history and a camera-motion prior. Its paper includes static-camera manipulation and identifies action conditioning as future work. It is not already an action-conditioned controller. Fast-WAM instead tests the benefit of video co-training with a direct action interface at inference. Their combination offers a hypothesis, not an established transfer result. [P1][P2]

For training labels, LIBERO can render depth and segmentation and expose camera transforms. Convert the normalized depth buffer to metric camera depth using robosuite's conversion, then unproject with calibrated intrinsics. At 1.4.0 the conversion is z = near / (1 - d(1 - near/far)), with clipping planes scaled by model extent. Transform depth, intrinsics and poses consistently with image rotation, crop and resize. A geometrically correct point cloud with mismatched image preprocessing is still the wrong policy input. [L1][R6]

A fixed external camera gives little value to a future camera-motion extrapolator; GeoWM already omits its redundant anchor in its static-camera setting. Wrist-camera motion depends on executed robot motion and contacts. A future wrist viewpoint inferred from candidate actions is a prediction, not future ground truth. Never feed future measured simulator poses to the deployed policy while calling it RGB-only. [P1]

For action learning, fix metric scale and coordinate conventions using training data or declared current-frame calibration. Do not align a predicted point cloud to future test ground truth before producing actions. Keep horizons tied to control steps and audit any no-op removal or temporal downsampling in the training export: frame index alone may not represent the same elapsed time across datasets.

A useful progression is: current geometry as a control, future-geometry supervision without a test-time decoder, then optional compact predicted-geometry conditioning at inference. Training-only simulator depth is privileged supervision and should be disclosed. Test-time simulator depth is an RGB-D/oracle condition; RGB-derived geometry is a different condition with perception cost and errors. Keep these experiments separate.

Demonstration-state replay can provide offline depth labels without running an expensive policy. Prefer recorded states and their matching model XML/assets over assuming recorded actions reproduce exactly under a new MuJoCo version. First verify reconstructed RGB alignment. If a converted training dataset has discarded states/XML, recover the original demonstrations and establish frame/action alignment; do not assume the LeRobot export contains everything required.

The decisive result is improved closed-loop action success with controlled latency and data, particularly under viewpoint/layout shifts. Lower future depth error alone does not establish a stronger policy. See the companion plan for ablations and a deadline-aware schedule.

## Sources

Core reading and implementation entry points:

- [GeoWM paper][P1], [Fast-WAM paper][P2], and [LIBERO-Plus arXiv v2][P3].
- [LIBERO environment wrapper][L1], [state stepping and reset][L2], [goal evaluation][L3], [predicates][L5], and [object-state semantics][L6].
- [OSC controller][R2], [controller scaling][R3], [state serialization][R5], and [camera geometry helpers][R6].
- [FastWAM evaluator][F1], [evaluation configuration][F3], and [OpenVLA-OFT comparison evaluator][O1].
- [Plus classification manifest][PLUS2], [corruption wrapper][PLUS3], and [variant initial-state resolution][PLUS4].
- [MuJoCo reproducibility][M1] and [PyTorch reproducibility][T1].

[L1]: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py
[L2]: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/bddl_base_domain.py
[L3]: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/problems/libero_tabletop_manipulation.py
[L4]: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/bddl_files/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl
[L5]: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/predicates/base_predicates.py
[L6]: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/object_states/base_object_states.py
[L7]: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/README.md
[L8]: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/lifelong/evaluate.py
[R1]: https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/environments/base.py
[R2]: https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/osc.py
[R3]: https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/config/osc_pose.json
[R4]: https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/macros.py
[R5]: https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/utils/binding_utils.py
[R6]: https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/utils/camera_utils.py
[F1]: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/libero/eval_libero_single.py
[F2]: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/libero/libero_utils.py
[F3]: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/sim_libero.yaml
[F4]: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/libero/run_libero_manager.py
[F5]: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/pyproject.toml
[F6]: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py
[F7]: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/utils/pytorch_utils.py
[O1]: https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/experiments/robot/libero/run_libero_eval.py
[PLUS1]: https://github.com/sylvestf/LIBERO-plus/blob/4976dc30028e805ff8094b55501d532c48fec182/README.md
[PLUS2]: https://github.com/sylvestf/LIBERO-plus/blob/4976dc30028e805ff8094b55501d532c48fec182/libero/libero/benchmark/task_classification.json
[PLUS3]: https://github.com/sylvestf/LIBERO-plus/blob/4976dc30028e805ff8094b55501d532c48fec182/libero/libero/envs/env_wrapper.py
[PLUS4]: https://github.com/sylvestf/LIBERO-plus/blob/4976dc30028e805ff8094b55501d532c48fec182/libero/libero/benchmark/__init__.py
[P1]: https://arxiv.org/html/2610.07381v1
[P2]: https://arxiv.org/html/2603.16666v1
[P3]: https://arxiv.org/html/2510.13626v2
[M1]: https://mujoco.readthedocs.io/en/stable/computation/index.html#reproducibility
[T1]: https://docs.pytorch.org/docs/2.14/notes/randomness.html
