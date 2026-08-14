# 최종 발표 대본 — 교정본

기준 자료: `outputs/0814.pdf` 25 pages. 학습 과정 내부 구간은 모두
**phase**라고 부르고, 전체 반복 단위만 **training step**이라고 부른다.

## Slide 1 — 제목

안녕하세요. 이화여자대학교 이주원입니다.

이번 여름방학에는 RL system 관련 논문을 읽고, 관심 있는 방향을 정해 직접
구현하고 실험했습니다. 오늘은 제한된 단일 GPU에서 full-parameter RL training을
가능하게 하기 위해 memory object의 lifetime을 phase에 맞춰 관리한 결과를
발표하겠습니다.

## Slide 2 — Background

먼저 실험을 이해하는 데 필요한 LLM 학습과 RL 학습 흐름을 간단히 설명한 뒤,
연구 질문과 실험 결과를 말씀드리겠습니다.

## Slide 3 — LLM Training Phases

LLM 학습은 크게 pretraining, instruction alignment, preference alignment의 세
phase로 나눌 수 있습니다. RL은 주로 마지막 preference alignment phase에서
사용됩니다.

Pretraining에서는 label이 없는 대규모 문장으로 다음 token을 예측하며 기본적인
언어 생성 능력을 학습합니다. Instruction alignment에서는 질문과 적절한 답변이
label된 데이터로 사용자의 instruction에 맞는 응답을 학습합니다. Preference
alignment에서는 생성된 답변이 인간의 선호 또는 주어진 평가 기준에 더 잘
부합하도록 모델을 조정합니다.

전통적인 RLHF에서는 인간의 선호를 대신 평가할 Reward Model을 먼저 학습합니다.
반면 수학 문제처럼 정답을 직접 판별할 수 있으면 별도의 Reward Model 대신
함수로 reward를 계산할 수 있습니다.

## Slide 4 — Basic RLHF Training Loop

회색 모델은 지금까지 학습한 LLM이며 RL에서는 Actor Model이라고 부릅니다.
Actor에 prompt를 입력하면 response가 생성되고, Reward Model 또는 reward
function이 이 response를 평가합니다. 이후 reward를 높이는 방향으로 Actor
parameter를 update하는 것이 기본적인 RL 학습 흐름입니다.

## Slide 5 — Why Use a Reference Model?

Reward만 높이도록 Actor를 계속 update하면 초기 model에서 지나치게 멀어질 수
있습니다. 이를 막기 위해 RL 시작 전의 초기 model을 고정한 Reference Model을
사용합니다.

Actor와 Reference의 출력 분포 차이가 커질수록 penalty를 부여해 Actor가 너무
급격히 변하지 않도록 제한합니다. 계산된 reward와 penalty를 바탕으로 더 좋은
response를 생성하도록 Actor parameter를 계속 update합니다.

## Slide 6 — GRPO Step as Multiple Phases

하나의 GRPO training step은 여러 phase로 구성됩니다.

먼저 Actor가 response를 생성하는 Rollout을 수행하고 reward를 계산합니다. 이후
Actor와 Reference의 log-probability를 계산해 penalty를 반영합니다. 마지막으로
Actor forward와 backward로 gradient를 계산하고 Update phase에서 parameter를
갱신합니다.

## Slide 7 — Research Question

RL training에는 Actor와 Reference, 경우에 따라 Reward Model까지 여러 model이
등장합니다. Parameter, gradient, optimizer state 같은 memory object도 함께
필요하므로 supervised fine tuning보다 GPU memory 요구량이 커질 수 있습니다.

여기서 연구 질문을 세웠습니다. 큰 GPU나 여러 GPU가 없는 환경에서도
full-parameter RL training을 가능하게 만들 수 있을까?

## Slide 8 — Memory Objects in GRPO

모든 memory object가 모든 phase에서 동시에 필요한 것은 아닙니다.

Actor parameter는 Rollout, Actor log-probability, forward, backward와 Update에
필요합니다. Reference parameter는 Reference log-probability에만 필요합니다.
Gradient는 backward 중 순차적으로 생성되어 Update 전까지 유지됩니다. Adam
state는 parameter Update에 사용되는 1차·2차 moment이며, FP32에서는 parameter
크기의 두 배입니다.

Qwen2.5-0.5B-Instruct에서는 Actor, Reference, gradient가 각각 1.843 GiB이고
Adam state는 3.686 GiB입니다.

## Slide 9 — Memory Liveness across GRPO Phases

표의 색칠된 원은 현재 phase의 GPU 연산에 직접 참여하는 object, 흰 원은 현재
연산에는 참여하지 않지만 이후를 위해 GPU에 유지된 object, 대시는 존재하지
않아도 되는 object입니다.

이 표를 보면 memory object의 실제 사용 시점이 서로 다르다는 것을 알 수
있습니다.

## Slide 10 — Fixed GPU Placement Is Wasteful

모든 memory object를 GPU에 계속 상주시키면 phase 전환은 빠르지만 현재 사용하지
않는 object도 제한된 GPU memory를 차지합니다. 특히 Reference와 Adam state는
사용하는 phase가 제한적인데도 fixed placement에서는 계속 GPU에 남습니다.

## Slide 11 — Phase Offloading

이를 해결하기 위해 현재 phase의 GPU 연산에 필요한 active memory object만 GPU에
두고, 나머지는 CPU로 offload하는 Phase Offloading을 적용했습니다.

Reference는 Reference log-probability phase에만 GPU로 가져오고, Adam state는
GPU AdamW Update phase에만 가져옵니다. 이 방식은 memory object의 lifetime을
GRPO 실행 phase와 맞추는 방법입니다.

## Slide 12 — Experimental Setup

11.90 GiB usable capacity의 NVIDIA TITAN Xp 단일 GPU에서 실험했습니다. 기본
모델은 Qwen2.5-0.5B-Instruct, dataset은 정답을 함수로 평가할 수 있는 GSM8K이며
별도 Reward Model은 사용하지 않았습니다. LoRA가 아닌 full-parameter GRPO를
실행했고 Rollout은 Hugging Face eager backend를 사용했습니다.

최종 성능 비교는 구성별 세 번 반복하고 앞의 두 training steps를 warm-up으로
제외했습니다. GPU/CPU AdamW 비교와 1.5B 성공 run은 30개 measured steps,
streaming 최종 비교는 28개 measured steps입니다.

## Slide 13 — Experimental Results

이제 직접 실행한 실험 결과를 설명드리겠습니다. 전체 흐름은 Phase Offloading,
CPU AdamW, gradient streaming 순서이며, 각 방법은 앞 방법을 적용한 뒤 새롭게
나타난 peak를 해결합니다.

## Slide 14 — Phase Offloading Recap

첫 실험은 현재 phase에서 사용하지 않는 Actor, Reference와 Adam state를 CPU로
offload한 결과입니다.

## Slide 15 — Phase Offloading Reduces Memory

그래프는 phase별 peak allocated GPU memory입니다. Phase Offloading을 적용하면
Rollout부터 Actor forward까지 memory peak가 크게 감소하고 Actor backward도
10.24 GiB에서 4.72 GiB로 감소합니다.

## Slide 16 — Update Peak Remains

하지만 전체 training step의 최대 peak를 보면 Update가 여전히 8.40 GiB입니다.
All-on-GPU의 10.24 GiB에서 줄어든 1.84 GiB는 Update에서 사용하지 않는 Reference
parameter의 크기와 일치합니다.

## Slide 17 — GPU AdamW Bottleneck

GPU AdamW Update에는 Actor parameter, 전체 gradient와 3.686 GiB Adam state가
동시에 필요합니다. 따라서 Phase Offloading은 사용하지 않는 residency는
효과적으로 제거하지만 Update 연산 자체가 요구하는 memory peak는 해결하지
못합니다.

단일 GPU에서 학습 가능 여부를 결정하려면 앞 phase의 감소보다 전체 training
step의 최대 peak를 줄이는 것이 중요합니다.

## Slide 18 — CPU AdamW

Update peak를 줄이기 위해 CPU AdamW를 도입했습니다.

GPU AdamW는 Update를 GPU에서 실행하므로 gradient와 Adam state가 GPU에 있어야
합니다. CPU AdamW는 parameter update를 CPU에서 수행하므로 FP32 master
parameter, gradient와 Adam state를 CPU에 둘 수 있습니다. 특히 큰 비중을
차지하는 Adam state를 GPU에서 제거할 수 있습니다.

## Slide 19 — CPU AdamW Trade-off

동일한 Phase Offloading 설정에서 optimizer 실행 위치만 비교했습니다.

GPU AdamW의 Update peak는 8.41 GiB지만 CPU AdamW는 2.37 GiB로 감소합니다.
반면 Update time은 0.13초에서 3.56초로 증가합니다. CPU AdamW는 memory를 크게
줄이지만 속도를 희생하는 trade-off가 있습니다.

## Slide 20 — New Bottleneck: Actor Backward

전체 phase를 살펴보면 CPU AdamW 적용 후 최대 peak가 Update에서 Actor
backward로 이동합니다. Update peak는 제거했지만 Actor parameter와 전체
gradient가 함께 존재하는 backward의 4.72 GiB가 새로운 bottleneck입니다.

## Slide 21 — Gradient Streaming

Backward는 GPU에서 진행하므로 Actor parameter는 GPU에 있어야 합니다. 대신
gradient는 layer별로 순차 생성된다는 점을 이용할 수 있습니다.

No-stream 방식은 모든 gradient가 만들어진 뒤 Update에서 전체 gradient를
CPU로 보냅니다. Gradient streaming은 일정 크기의 gradient가 준비될 때마다
bucket 단위로 CPU에 전송하고 GPU 원본을 조기에 해제합니다. 이를 통해 전체
gradient가 GPU에 동시에 상주하는 것을 피합니다.

이 아이디어는 ZeRO-Offload의 gradient offloading을 GRPO의 phase-aware 실행에
적용한 것입니다.

## Slide 22 — Gradient Streaming Result

최종 16 MiB, 3-slot gradient streaming은 Actor-backward peak를 4.72 GiB에서
3.40 GiB로 줄였습니다.

Backward time은 gradient packing, D2H 전송과 staging-slot backpressure 때문에
0.21초에서 0.29초로 증가했습니다. 하지만 gradient를 CPU에 미리 준비하고 CPU
Adam bucket update와 updated-parameter H2D를 overlap하면서 Update는 3.55초에서
3.02초로 감소했습니다.

결과적으로 전체 training step은 9.81초에서 8.63초로 12.1% 단축됐습니다.

## Slide 23 — 1.5B Capacity

최적화가 실제로 더 큰 model의 학습 가능 여부를 바꾸는지 Qwen2.5-1.5B FP32로
확인했습니다. Actor, Reference와 gradient는 각각 5.751 GiB이고 Adam state는
11.502 GiB입니다. 이 model-state를 단순 합산하면 약 28.755 GiB로 GPU capacity
11.90 GiB를 크게 넘습니다.

All GPU는 OOM이 발생했습니다. CPU AdamW만 적용한 no-stream 구성도 OOM이
발생했고 CPU AdamW와 gradient streaming을 함께 적용한 구성만 8.44 GiB phase
peak로 성공했습니다.

## Slide 24 — OOM before Update

CPU AdamW no-stream 구성은 Update에 도달하기 전에 Actor backward에서 OOM이
발생했습니다. OOM 시 PyTorch allocated memory는 11.34 GiB였고 추가 54 MiB
allocation에 실패했습니다.

즉 optimizer state를 CPU로 옮겼지만 backward 중 full gradient가 누적되면서
새로운 capacity bottleneck이 된 것입니다.

## Slide 25 — Conclusion

CPU AdamW는 Update에서 GPU optimizer-state residency를 제거하고, gradient
streaming은 backward의 full-gradient residency를 제거합니다. 두 방법은 같은
memory를 줄이는 것이 아니라 서로 다른 peak 원인을 해결합니다.

Phase Offloading, CPU AdamW와 gradient streaming을 결합해 model-state 단순
합계가 최대 28.755 GiB인 Qwen2.5-1.5B FP32 full-parameter GRPO를 11.90 GiB
단일 GPU에서 8.44 GiB phase peak로 실행했습니다.

이 실험을 통해 memory object의 lifetime을 training phase에 맞춰 관리하면,
속도와 transfer 비용의 trade-off는 있지만 제한된 단일 GPU에서도 더 큰 RL
workload를 실행할 수 있음을 확인했습니다.
