# VERL Single-GPU Offloading Study: Final C, G, and M Results

## 1. Scope and research questions

This report consolidates the final **C, G, and M series** for single-GPU, full-parameter VERL GRPO training. Each series answers a different question.

- **C series — forward ablation:** Starting from an all-GPU configuration, what happens when Reference parameters, AdamW optimizer state, or saved activations are offloaded individually?
- **G series — reverse residency ablation:** With Actor parameters kept resident on the GPU to avoid confounding the experiment with Actor swapping, which additional object should be prioritized for GPU residency?
- **M series — optimizer execution placement:** What memory reduction and runtime penalty result when AdamW computation itself, rather than only its state, is moved to the CPU?

The three series must be interpreted together. C measures the cost of introducing offload from an all-GPU endpoint. G verifies the same placement choices in the reverse direction while holding Actor residency fixed. M then changes the optimizer execution device and exposes the next memory bottleneck after GPU AdamW pressure is removed.

The final analysis intentionally excludes the broader C combinations and allocator-diagnostic A series from the main result set. Those runs remain useful diagnostics, but the final policy claims below are based on C00–C03, G00–G03, and M00–M02.

## 2. Experimental setup and metric definitions

| Item | Setting |
|---|---|
| Framework | VERL single-controller GRPO with Ray workers and PyTorch FSDP1 |
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, full-parameter training (`LoRA rank=0`) |
| Dataset | GSM8K |
| Hardware | One GPU with approximately 11.9 GiB of memory |
| Batch | train batch 2, PPO mini-batch 2, micro-batch/GPU 1 |
| Sequence | maximum prompt length 128, maximum response length 64, rollout samples per prompt `n=2` |
| Repetition protocol | 30 training steps per run, 3 independent runs per configuration, first 5 steps excluded |
| Precision observed in detailed M instrumentation | FSDP flat Actor parameter and gradient storage in FP32 |
| Main metrics | mean step time, tokens/s/GPU, peak PyTorch CUDA allocation, phase time, phase peak allocation |

The `(n=3)` shown below each graph label means **three repeated runs**, not three rollout samples. The rollout configuration remains `n=2`.

### Metric boundaries

- **Step time** is the end-to-end mean training-step duration after warm-up.
- **Throughput** is tokens per second per GPU; higher is better.
- **Peak allocated GPU memory** is the maximum live CUDA allocation tracked by PyTorch. It is the primary tensor-memory metric.
- **GPU device-used memory** additionally includes CUDA context, allocator-reserved blocks, libraries, and other driver-visible allocations. It can remain high after live tensors have been moved to the CPU.
- A phase-level `Actor update` peak covers the entire Actor forward/backward and optimizer sequence. It does **not** mean that the peak occurred inside the AdamW arithmetic kernel.
- Phase values are means over the three successful repetitions. Near-zero memory standard deviations are expected because tensor shapes and allocation paths are deterministic under this fixed workload.

Cross-series absolute times should be compared cautiously because C, G, and M were collected in separate experiment batches. The strongest causal comparisons are within each series; cross-series conclusions rely mainly on repeated qualitative agreement.

## 3. C series: forward offloading ablation

The Actor remains GPU-resident in all four C configurations. AdamW computation remains on the GPU. Each non-baseline configuration changes one object relative to C00.

| ID | Reference parameters | Optimizer state | Saved activations | Actor parameters | Interpretation |
|---|---|---|---|---|---|
| C00 | GPU resident | GPU resident | GPU | GPU resident | all-GPU control |
| C01 | phase-offloaded to CPU | GPU resident | GPU | GPU resident | Reference offload only |
| C02 | swapped CPU↔GPU by phase | CPU between uses | GPU | GPU resident | optimizer-state offload only |
| C03 | GPU→CPU saved-tensor offload | GPU resident | CPU-backed | GPU resident | activation offload only |

### 3.1 End-to-end results

| ID | Step time (s) | Throughput (tokens/s/GPU) | Global GPU peak (GiB) | CPU RSS peak (GiB) | Change from C00 |
|---|---:|---:|---:|---:|---|
| C00 | 6.656 | 100.92 | 11.065 | 1.237 | baseline |
| C01 | 6.923 | 97.06 | 10.149 | 2.486 | GPU −0.916 GiB; step +4.0% |
| C02 | 7.505 | 89.39 | 11.065 | 5.233 | no global-peak reduction; step +12.8% |
| C03 | 7.146 | 94.09 | 11.066 | 1.762 | no global-peak reduction; step +7.4% |

### 3.2 Phase execution time

| Configuration | Rollout (s) | Actor log-prob (s) | Reference log-prob (s) | Actor update (s) |
|---|---:|---:|---:|---:|
| C00 | 4.562 | 0.321 | 0.311 | 1.452 |
| C01 | 4.548 | 0.322 | 0.568 | 1.472 |
| C02 | 4.987 | 0.325 | 0.325 | 1.857 |
| C03 | 4.559 | 0.326 | 0.328 | 1.923 |

### 3.3 Phase peak allocated GPU memory

| Configuration | Rollout (GiB) | Actor log-prob (GiB) | Reference log-prob (GiB) | Actor update (GiB) |
|---|---:|---:|---:|---:|
| C00 | 7.409 | 7.687 | 7.021 | 11.064 |
| C01 | 6.507 | 6.767 | 7.021 | 10.149 |
| C02 | 3.728 | 4.007 | 3.340 | 11.064 |
| C03 | 7.409 | 7.687 | 7.021 | 11.066 |

### 3.4 Interpretation

#### Reference offload is the only C-series option that reduces the global peak

C01 lowers the global allocated peak from 11.065 to 10.149 GiB, a 0.916 GiB reduction. It also lowers Rollout and Actor log-prob peaks by approximately 0.90–0.92 GiB because the inactive Reference copy no longer occupies GPU memory during those phases. Reference log-prob itself still reaches 7.021 GiB because the Reference model must be loaded for its forward pass.

The cost is localized and easy to explain: Reference log-prob time increases from 0.311 to 0.568 s due to parameter loading and offloading. The other phases remain nearly unchanged. Reference offload is therefore a predictable memory-for-time exchange.

#### Optimizer-state offload reduces non-update residency but not the global peak

C02 dramatically lowers phase peaks outside Actor update: Rollout falls from 7.409 to 3.728 GiB, Actor log-prob from 7.687 to 4.007 GiB, and Reference log-prob from 7.021 to 3.340 GiB. However, the optimizer state must return to the GPU for GPU AdamW, so Actor update still peaks at 11.064 GiB. Consequently, the global peak is unchanged.

Its runtime cost is also substantial. Actor update increases from 1.452 to 1.857 s, and Rollout increases from 4.562 to 4.987 s. The measured configuration therefore pays state-transfer and residency-transition overhead without solving the maximum-memory phase.

#### Activation offload does not reduce this workload's measured peak

C03 leaves every measured phase peak effectively equal to C00 while increasing Actor update from 1.452 to 1.923 s. Saved-tensor offload moves activations only after they are produced and retrieves them for backward. Current-layer activations, gradients, parameters, logits, and transient workspaces still coexist at compute time. Transfer can also temporarily overlap source and destination lifetimes. Therefore, saved-activation offload is not equivalent to eliminating the Actor F/B working set.

This result does not prove that activation offload is universally useless. It shows that, for this short-sequence, micro-batch-one workload, it provides no measurable peak reduction while imposing a clear time penalty.

## 4. G series: Actor-resident reverse residency ablation

The final G series uses the original **Actor-resident** baseline. This is important: Actor parameters are held constant on the GPU in G00–G03, so the ablation does not grow by adding Actor residency as a separate variable. AdamW remains a GPU operation throughout.

G00 offloads Reference parameters, swaps optimizer state, and offloads saved activations. G01–G03 each keep exactly one of those objects on the GPU relative to G00.

| ID | Actor | Reference | Optimizer state | Saved activations | Object added to GPU relative to G00 |
|---|---|---|---|---|---|
| G00 | GPU resident | phase-offloaded | CPU between uses | CPU-backed | none; Actor-resident base |
| G01 | GPU resident | GPU resident | CPU between uses | CPU-backed | Reference parameters |
| G02 | GPU resident | phase-offloaded | GPU resident | CPU-backed | optimizer state |
| G03 | GPU resident | phase-offloaded | CPU between uses | GPU | saved activations |

### 4.1 End-to-end results

| ID | Step time (s) | Throughput (tokens/s/GPU) | Global GPU peak (GiB) | GPU device peak (GiB) | CPU RSS peak (GiB) | Change from G00 |
|---|---:|---:|---:|---:|---:|---|
| G00 | 7.985 | 84.14 | 10.149 | 9.546 | 7.017 | baseline |
| G01 | 7.872 | 85.31 | 11.066 | 10.497 | 5.763 | step −1.4%; GPU +0.917 GiB |
| G02 | 7.330 | 91.20 | 10.149 | 10.292 | 3.018 | step −8.2%; no allocated-peak increase |
| G03 | 7.584 | 88.63 | 10.149 | 9.544 | 6.486 | step −5.0%; no allocated-peak increase |

### 4.2 Phase execution time

| Configuration | Rollout (s) | Actor log-prob (s) | Reference log-prob (s) | Actor update (s) |
|---|---:|---:|---:|---:|
| G00 | 4.948 | 0.314 | 0.559 | 2.155 |
| G01 | 4.922 | 0.322 | 0.321 | 2.298 |
| G02 | 4.529 | 0.326 | 0.581 | 1.885 |
| G03 | 4.856 | 0.322 | 0.571 | 1.824 |

### 4.3 Phase peak allocated GPU memory

| Configuration | Rollout (GiB) | Actor log-prob (GiB) | Reference log-prob (GiB) | Actor update (GiB) |
|---|---:|---:|---:|---:|
| G00 | 2.826 | 3.086 | 3.340 | 10.149 |
| G01 | 3.728 | 4.007 | 3.340 | 11.066 |
| G02 | 6.507 | 6.767 | 7.021 | 10.149 |
| G03 | 2.826 | 3.086 | 3.340 | 10.149 |

### 4.4 Interpretation

#### Reference residency buys little performance and directly raises the peak

G01 reduces Reference log-prob time from 0.559 to 0.321 s because the Reference parameters no longer need to be transferred. Nevertheless, Actor update becomes 0.143 s slower, and the end-to-end gain is only 1.4%. At the same time, the global peak rises by 0.917 GiB to 11.066 GiB.

The phase-memory pattern is the reverse of C01: keeping Reference parameters resident adds approximately 0.90–0.92 GiB to Rollout, Actor log-prob, and Actor update. C01 and G01 therefore agree: Reference parameters are the first object to offload when memory is constrained.

#### Optimizer residency gives the largest speedup when memory headroom exists

G02 is the fastest G configuration. It improves step time by 8.2% and throughput by approximately 8.4% relative to G00. The main gains occur in Rollout and Actor update, consistent with removing optimizer load/offload transitions and avoiding the GPU-AdamW state swap.

Optimizer residency substantially increases the non-update phase allocations—roughly 3.68 GiB, matching the two FP32 Adam moment buffers—but it does not raise the global allocated peak because G00 already pays that state footprint when the optimizer is loaded for Actor update. This is a crucial distinction between **residency** and **peak requirement**: keeping a state resident can increase memory throughout the step without increasing the maximum if the same state is required at the existing peak.

#### Keeping activations on GPU is faster and does not increase the measured peak

G03 improves step time by 5.0%, primarily through Actor update decreasing from 2.155 to 1.824 s. Its phase peak allocations are effectively identical to G00. This is fully consistent with C03: activation offload adds pack/copy/unpack overhead but fails to reduce the phase maximum in this workload.

Therefore, under the tested conditions, saved activations should remain on the GPU. This conclusion is workload-specific and should be revalidated for longer sequences or larger micro-batches, where saved activations may become a larger fraction of the working set.

## 5. M series: CPU AdamW and the exposed Actor F/B bottleneck

The M series changes more than optimizer-state residency. In M00 and M01, the FSDP flat Actor parameters and gradients are moved to the CPU and standard PyTorch AdamW executes there. The updated Actor parameters are then copied back to the GPU. M02 is the matched Actor-resident GPU-Adam control.

The accurate presentation names are:

- **M00 — CPU Adam / Partial actor reuse:** retain Actor parameters only across Rollout→Actor log-prob, then phase-offload them elsewhere; CPU AdamW.
- **M01 — CPU Adam / Actor resident:** keep Actor parameters on GPU across RL phases, except for the explicit CPU AdamW step where the implementation moves them to the CPU.
- **M02 — GPU Adam / Actor resident:** keep Actor parameters resident and execute AdamW on the GPU, with optimizer-state swapping.

| ID | Actor residency across RL phases | Reference | AdamW device | Optimizer state | Saved activations |
|---|---|---|---|---|---|
| M00 | partial reuse, otherwise phase-offload | phase-offload | CPU | CPU | CPU-backed |
| M01 | GPU resident outside CPU optimizer step | phase-offload | CPU | CPU | CPU-backed |
| M02 | GPU resident | phase-offload | GPU | CPU between uses | CPU-backed |

### 5.1 End-to-end results

| ID | Step time (s) | Throughput (tokens/s/GPU) | Global GPU peak (GiB) | GPU device peak (GiB) | CPU RSS peak (GiB) |
|---|---:|---:|---:|---:|---:|
| M00 | 12.979 | 51.76 | 6.469 | 8.972 | 8.667 |
| M01 | 12.307 | 54.26 | 6.468 | 8.359 | 8.670 |
| M02 | 8.226 | 81.73 | 10.151 | 10.816 | 7.010 |

Relative to M02, M00 lowers peak allocation by 3.682 GiB (36.3%) but increases step time by 57.8%. M01 lowers peak allocation by 3.683 GiB (36.3%) while increasing step time by 49.6%. M01 is 5.2% faster than M00 at effectively the same allocated peak.

### 5.2 Phase execution time

| Configuration | Rollout (s) | Actor log-prob (s) | Reference log-prob (s) | Actor update (s) |
|---|---:|---:|---:|---:|
| M00 | 4.763 | 0.487 | 0.571 | 7.145 |
| M01 | 4.583 | 0.314 | 0.575 | 6.823 |
| M02 | 4.904 | 0.325 | 0.583 | 2.405 |

### 5.3 Phase peak allocated GPU memory

| Configuration | Rollout (GiB) | Actor log-prob (GiB) | Reference log-prob (GiB) | Actor update (GiB) |
|---|---:|---:|---:|---:|
| M00 | 2.823 | 3.085 | 1.499 | 6.469 |
| M01 | 2.810 | 3.086 | 3.340 | 6.468 |
| M02 | 2.826 | 3.086 | 3.340 | 10.151 |

### 5.4 What CPU AdamW actually removes

Detailed M00 instrumentation at step 10 records the following placement transition:

| Snapshot | CUDA allocated (GiB) | Actor parameter | Gradient | CPU Adam states |
|---|---:|---|---|---|
| End of Actor backward | 3.697 | GPU 1.840 GiB | GPU 1.840 GiB | CPU 3.681 GiB |
| After parameter/gradient D2H | 0.016 | CPU 1.840 GiB | CPU 1.840 GiB | CPU 3.681 GiB |
| After CPU AdamW | 0.016 | CPU 1.840 GiB | CPU 1.840 GiB | CPU 3.681 GiB |
| After Actor reload | 1.856 | GPU 1.840 GiB | released | CPU 3.681 GiB |

This confirms that CPU AdamW itself is not responsible for the remaining 6.47 GiB phase peak. During the actual CPU optimizer calculation, live CUDA tensor allocation falls to approximately 16 MiB.

### 5.5 Why the Actor-update peak remains 6.47 GiB

The `Actor update` bar is the maximum over the complete update phase. After GPU AdamW pressure is removed, the maximum shifts earlier to Actor forward/backward, where the GPU simultaneously requires:

- Actor parameters;
- gradients produced by backward;
- current and saved activations required by autograd;
- logits, loss tensors, attention workspaces, and transient operator buffers.

CPU AdamW changes only optimizer placement and execution. It cannot remove the working set required to compute Actor forward/backward. Thus M00/M01 solve the optimizer-state peak and expose Actor F/B as the next bottleneck.

### 5.6 Why CPU AdamW is slow in the current implementation

The current implementation is synchronous and whole-model based:

1. complete Actor backward on the GPU;
2. move the entire FSDP flat Actor parameter storage to the CPU;
3. synchronize the device;
4. move all gradients synchronously to the CPU;
5. execute standard PyTorch AdamW on the CPU;
6. release CPU gradients;
7. reload the entire updated Actor parameter storage to the GPU.

There is no gradient streaming during backward, bucket-level overlap, optimized DeepSpeed-style CPU Adam kernel, double buffering, or one-step-delayed optimizer overlap. The measured slowdown therefore represents a **naive synchronous CPU-Adam endpoint**, not the performance ceiling of an optimized ZeRO-Offload implementation.

M01 outperforms M00 because keeping Actor parameters resident across the surrounding RL phases removes avoidable Actor swaps. It does not eliminate the mandatory parameter D2H/H2D transfer inside the current CPU optimizer step.

## 6. Combined interpretation of C, G, and M

### 6.1 C and G provide consistent bidirectional evidence

The two ablations tell the same story from opposite directions.

| Object | C: offload from all-GPU | G: keep resident from Actor-resident offload base | Combined conclusion |
|---|---|---|---|
| Reference parameters | saves 0.916 GiB globally for +4.0% step time | costs 0.917 GiB for only 1.4% speedup | first persistent object to offload |
| Optimizer state | no global-peak reduction; +12.8% step time | 8.2% faster at the same global peak | keep resident when memory headroom exists; otherwise offload for lower inter-phase residency, not lower GPU-Adam peak |
| Saved activations | no peak reduction; +7.4% step time | 5.0% faster with no peak increase | keep on GPU for this workload |

The optimizer result is not contradictory. Offloading optimizer state lowers memory in Rollout and log-prob phases, but GPU AdamW still needs it during Actor update, so it cannot lower the global maximum. Conversely, keeping it resident eliminates transfers without increasing the already optimizer-dominated maximum.

### 6.2 M changes the optimization regime

C and G retain GPU AdamW. Their global peak is approximately 10.15–11.07 GiB and is dominated by Actor update with optimizer state present. M00/M01 move the AdamW computation and state to the CPU, reducing the peak to approximately 6.47 GiB. The dominant object then changes from the GPU optimizer working set to the Actor F/B working set.

This produces a two-stage bottleneck interpretation:

1. **GPU-Adam regime:** optimizer states and AdamW temporaries dominate Actor-update memory.
2. **CPU-Adam regime:** optimizer pressure disappears, and Actor parameter + gradient + activation + transient F/B memory becomes the peak.

### 6.3 Recommended static policy under the measured workload

For the tested model, sequence length, and micro-batch:

1. **Keep Actor parameters resident across RL phases.** They are reused by Rollout, Actor log-prob, and Actor update; M01 is faster than partial Actor reuse at the same CPU-Adam memory peak.
2. **Offload Reference parameters first when memory is constrained.** They are read-only, used in one phase, and provide the clearest global memory saving per unit of runtime cost.
3. **Keep saved activations on GPU under this workload.** Offload does not lower the observed peak and slows Actor update.
4. **Prioritize optimizer-state GPU residency when sufficient memory is available.** G02 is the fastest G configuration and does not increase the global peak relative to G00, although it raises memory throughout non-update phases.
5. **Use CPU AdamW only when the approximately 3.68 GiB peak reduction is necessary.** Among the measured CPU-Adam choices, M01 is preferable to M00.

The resulting priority order is:

> Actor residency → activation residency → optimizer residency when capacity permits → Reference offload first.

This ordering is conditional on the current workload. For longer sequences or larger micro-batches, activation memory may grow enough to change the policy.

## 7. Limitations and claims that should not be made

- These are single-GPU FSDP1 results on a 0.5B model and approximately 11.9 GiB GPU. They do not establish multi-GPU scaling behavior.
- M00/M01 use standard synchronous PyTorch CPU AdamW, not the fully optimized ZeRO-Offload schedule. Their runtime should not be presented as the inherent cost of all CPU-optimizer implementations.
- Activation conclusions apply to the measured prompt/response lengths and micro-batch. They should not be generalized to long-context training without another sweep.
- Separate experiment batches show modest absolute timing drift. Causal claims should use matched configurations inside each series.
- CUDA `allocated`, CUDA `reserved`, and driver-visible device usage are different quantities. A high device-used value during CPU AdamW does not imply that Actor tensors remain on the GPU.
- The G series used here is specifically the Actor-resident G00–G03 dataset from `offload-gpu-adam-residency-v1-performance`, matching the supplied final graph. It must not be mixed with the later CPU-residency G redesign.

## 8. Final conclusions

1. Offloading is not monotonically beneficial: removing more objects from the GPU can add transfer time without reducing the phase that determines the global maximum.
2. Reference parameters are the strongest first offload candidate. C01 and G01 independently show an approximately 0.92 GiB memory cost for Reference residency and only modest performance benefit.
3. Optimizer-state residency is a performance optimization when GPU capacity permits. It removes repeated state movement and gives the largest G-series speedup, but optimizer-state offload alone cannot reduce a GPU-AdamW global peak.
4. Saved-activation offload is counterproductive in the tested workload: it adds Actor-update overhead without reducing measured peak allocation.
5. Moving AdamW computation to the CPU is the only measured method that removes the GPU optimizer working set and reduces peak allocation from about 10.15 to 6.47 GiB.
6. Once GPU AdamW is removed, Actor forward/backward becomes the new bottleneck. Further memory reduction requires F/B-specific techniques such as activation recomputation, smaller micro-batches, gradient streaming, or layer/bucket-level parameter scheduling.
7. The current CPU-Adam implementation pays a large synchronous transfer and CPU-compute penalty. A natural next systems step is bucketized gradient D2H, CPU update, and updated-parameter H2D with overlap, while preserving Actor residency across RL phases.

## 9. Final result sources

- C: `outputs/offload-fullft-v5-performance` — C00–C03
- G: `outputs/offload-gpu-adam-residency-v1-performance` — Actor-resident G00–G03 shown in the final supplied graph
- M: `outputs/offload-residency-v2-performance` — M00–M02
- M detailed placement evidence: `outputs/offload-residency-v1-detail/M00_phase_min-r1/events/memory-actor_update_detail-rank0-pid1232276.jsonl`

Only successful runs are included in means. Every configuration in the final C00–C03, G00–G03, and M00–M02 set has three successful repetitions.
