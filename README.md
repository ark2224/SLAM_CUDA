# Approach to Robust SLAM in GPS‑Denied, Dynamic Environments

**Candidate:** Drew Kalasky
**Role:** Computer Vision Engineer, Maneuver Dominance
**Focus:** Real‑time, resilient 3D perception for autonomous aerial platforms

## Problem Context

Aerial autonomy in contested environments can’t rely on GPS. Perception must deliver stable localization and mapping in the presence of high dynamics (rapid motion, vibration), dynamic agents (people/vehicles), occlusion, and sensor noise—while meeting tight tail‑latency budgets so planning and control never starve.

My work product pairs a clear, measurable SLAM plan with CUDA‑accelerated kernels and systems guardrails. The code intentionally avoids heavy external deps to highlight design decisions rather than framework plumbing.


## Approach

**What I built (and why)**

**Files:**  profiler.cu (1), frontend.cu (2), backend_pcg.cu (3), streaming.cu (4), state_bus.* (5), demo_main.cu (0)

Rebuild and Run: 

'''bash
  mkdir -p build && cd build
  cmake ..
  cmake --build . -j
  ./slam_cuda_demo
'''


1) **Measurement & Profiling — make performance first‑class**

- Goal: Quantify p50/p95 latency and utilization with reproducibility.

- Implementation: NVTX ranges (optional), CUDA events, percentile aggregation. Warmups and fixed inputs reduce variance.

- Why it matters: Maneuver Dominance needs bounded tail latency, not just fast averages. This forms the contract with downstream planners.

2) **Robust Front‑End — features that survive dynamics and clutter**

- Goal: Extract and validate stable 2D‑2D correspondences even with motion blur and moving objects.

- Implementation:

  - FAST + 3×3 NMS on GPU for dense, evenly distributed keypoints.

  - Semantic down‑weighting hook: scores can be attenuated via a per‑pixel mask (e.g., people/vehicles), minimizing drift from dynamic agents.

  - IMU‑prior epipolar filter: Use inertial delta pose to gate matches with a robust (Huber‑like) residual, cutting RANSAC time and outliers.

- Why it matters: Robust correspondences are the lifeblood of VO/SfM in rapid, aggressive flight profiles.

3) **Resilient Back‑End — incremental optimization that converges**

- Goal: Maintain accurate state under compute pressure and outliers.

- Implementation:

  - CSR SpMV + Jacobi‑PCG in CUDA as a stand‑in for normal‑equation solves in an incremental factor graph.

  - Numerically safe updates (guards on division; early exit on convergence) and clean separation of math ops (axpy, dot, scale).

- Why it matters: Even if the full graph lives on CPU (g2o/CERES/ISAM2), GPU‑accelerated linear algebra unlocks headroom for higher update rates.

4) **Real‑Time Optimization — hide transfers, shape memory, avoid stalls**

- Goal: Keep the perception loop on deadline despite heavy throughput.

- Implementation:

  - Pinned host buffers + pre‑allocated device memory to kill allocator jitter.

  - Non‑blocking stream and event fencing to overlap H↔D copies with kernels.

  - A thin driver that illustrates frame‑asynchronous behavior.

- Why it matters: Planners need consistent, bounded delivery—p95/p99 are the KPIs.

5) **Systems‑Level Integration — safe outputs for autonomy**

- Goal: Provide a planner‑friendly state with backpressure handling and a latency watchdog.

- Implementation:

  - SPSC ring for cross‑thread, lock‑free handoff of StateEstimate (pose, velocity, covariances, timestamp, seq).

  - Tail‑latency guard: sliding‑window p95 with a “degrade mode” signal (e.g., raise FAST threshold, skip descriptors) to protect control loops.

- Why it matters: It makes perception a reliable service in a larger aerial system, not just a standalone demo.

## Snapshot Metrics (demo ⇒ flight targets)
| Component | Demo Behavior | Flight Target (initial) |
|---|---|---|
| Frontend latency | p95 ~ **≤8–10 ms** | Maintain p95 < **10 ms** on rep. HW |
| Estimator | Converges on toy SPD; clean swap to factor graph | **Bounded drift** (< ~1% over short windows) |
| Recovery | IMU‑aided gating reduces RANSAC iterations | **Relocalize < 0.5 s** after loss |
| Robustness | Degrade mode protects control loop | No planner starvation under clutter/EW |


## SLAM Thought Process & Engineering Tradeoffs

- **Sensing & priors:** Fuse IMU orientation/velocity to stabilize epipolar checks before RANSAC; incorporate semantics to down‑weight dynamic agents without fully discarding texture.

- **Keypoint economy:** Prefer spatially regular features over sheer counts (grid/NMS). Quality > quantity; lowers matcher and back‑end load.

- **Latency over throughput:** Optimize for tail latency and deterministic memory patterns (pinned buffers, pre‑alloc, streams). Throughput follows.

- **Graceful degradation:** When the world or compute budget gets hard, reduce front‑end work (thresholds, per‑cell caps) to keep the loop alive.

- **Determinism:** Fixed warmups, seeds (where applicable), and reproducible replays → faster triage and more credible field performance.


## How this maps to Maneuver Dominance needs

Works in GPS‑denied regimes, with SLAM outputs that are consumable by autonomy.

Designed for larger, more capable UAVs under high dynamics: IMU‑aided geometry, robust front‑end, and GPU acceleration.

Emphasis on integration and stakeholder trust: telemetry (p50/p95), watchdogs, and reproducible benchmarking.

Example metrics & expected behavior

Front‑end on a high‑contrast pattern: tens of thousands of NMS peaks; stored cap (e.g., 2k) for even coverage.

PCG back‑end on toy SPD: converges to stable solution; drop‑in path to full factor‑graph solves.

Latency guard: p95 under 8–10 ms in demo; triggers degrade if exceeded.



## Next Steps if Onboarded

Swap toy math with flight‑proven geometry and ISAM‑style factor graphs.

Integrate real sensors (stereo/RGB‑D/LiDAR, IMU) and test on representative flight data.

Build closed‑loop autonomy trials, tune degrade policies, and publish reproducible benchmarks to perception & planning teams.

Extend GPU usage where it moves the needle (feature selection, sparse linear solves, robustification passes).


