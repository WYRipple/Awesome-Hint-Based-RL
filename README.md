<h1 align="center">💫 Awesome LLM Hint-based RLVR</h1>


<p align="center">
  <b>A curated collection of papers and resources on Hint-Based RLVR for Large Language Models.</b>
</p>

## 😳 What are Hints?

**Hints** are **explicit textual signals** that carry task-relevant information beyond what ordinary rollouts under the current policy can reliably produce. They are introduced into the RL training process to reshape rollout-group construction, making otherwise uninformative samples more likely to yield useful policy updates. 

## 🧐 Why Hints?

Group-relative advantage methods like GRPO rely on **reward variation** within each rollout group to produce a gradient signal. When all rollouts in a group receive identical rewards, whether all correct on easy problems or all wrong on hard ones, the advantage collapses to zero and the policy-gradient signal vanishes. This **zero-advantage failure** wastes both computation and valuable training data, especially the hardest questions that matter most for learning. 

While searching more aggressively within the policy's own distribution (e.g., resampling or tree search) can help, it only sharpens the existing distribution without **expanding its capability boundary**. 

Hints address this bottleneck by injecting **external textual signals** that restore learnable contrasts within rollout groups, enabling the policy to learn from otherwise wasted samples while pushing beyond its current capability frontier.

<p align="center">
  📖 <b>Survey Paper:</b> <a href="#">A Survey on Hint-Based RLVR: Overcoming Zero-Advantage Failures with External Textual Signals</a>
</p>

<p align="center">
  🌕️ = Covered in our <a href="#">survey paper</a> &nbsp;|&nbsp; 🌒 = Indexed here, pending inclusion in the next survey revision
</p>

## 🛸 Full Tree
```text
Hint-based Reinforcement Learning (Current Survey Structure)
│
├── §3 Sample-level Hints
│   │
│   ├── §3.1 Trajectory-based Hints
│   │   ├── §3.1.1 Trajectory Injection
│   │   │   ├── Reference Prefix
│   │   │   │         (QuestA, POPE, CCL, SEELE, GHPO, BREAD)
│   │   │   ├── Self-Replay Prefix
│   │   │   │         (HiPO, PROS, RPO, Failure-Prefix)
│   │   │   └── Hybrid Prefix
│   │   │             (CORE)
│   │   ├── §3.1.2 Trajectory Continuation
│   │   │   ├── Scheduled Prefix Decay
│   │   │   │         (UFT, Prefix-RFT, EvoCoT)
│   │   │   ├── Adaptive Prefix Control
│   │   │   │         (G²RPO-A, ADHint, Hint-GRPO, BHA, TRAPO)
│   │   │   └── Intra-group Prefix Mixing
│   │   │             (StepHint)
│   │   ├── §3.1.3 Trajectory Replacement
│   │   │   ├── Unconditional Augmentation
│   │   │   │         (ANCHOR, LUFFY)
│   │   │   ├── Triggered Substitution
│   │   │   │         (AMPO, S-GRPO, HAPO, HiPO)
│   │   │   └── Reconstructive Repair
│   │   │             (SCOPE)
│   │   └── §3.1.4 Trajectory Optimization
│   │             (ExPO, MENTOR)
│   │
│   └── §3.2 Scaffold-based Hints
│       ├── §3.2.1 Scaffold Injection
│       │   ├── Answer-level Scaffold
│       │   │         (RAVR, CoVRL, HDPO)
│       │   ├── Solution-blueprint Scaffold
│       │   │         (Guide-GRPO, A2D, PieceHint, SAGE_scaf, AVATAR, RLAD)
│       │   ├── Knowledge-level Scaffold
│       │   │         (KnowRL, NuRL)
│       │   └── Format-level Scaffold
│       │             (MeRF, RuscaRL)
│       ├── §3.2.2 Scaffold Replacement
│       │   ├── Pedagogical Guidance
│       │   │         (HINT, Scaf-GRPO, KEPO, HiLL)
│       │   ├── Critique-driven Refinement
│       │   │         (LTE, RGR-GRPO, R³L, GOLF, ECHO)
│       │   └── Action-level Guidance
│       │             (InfoFlow)
│       └── §3.2.3 Scaffold Optimization
│           ├── Auxiliary Supervision
│           │         (MEL, ThinkTuning, RLTF, KEPO)
│           └── Distribution Alignment
│                     (HDPO, RAVR, CoVRL)
│
├── §4 Task-level Hints
│   ├── §4.1 Static Experience Base
│   │   ├── Demonstration-based Experience
│   │   │         (CBRL, ICPO)
│   │   └── Skill-based Experience
│   │             (TemplateRL, SKILL0)
│   └── §4.2 Evolving Experience Base
│       ├── Accumulative Experience Base
│       │   ├── Reflection Experience
│       │   │         (EvolveR, RetroAgent, ERL, MAGE)
│       │   └── Strategy Experience
│       │             (SGE, CRMWeaver, IntPro, MetaClaw)
│       ├── Curated Experience Base
│       │   ├── Skill Evolution
│       │   │         (SkillRL, ARISE, COS-PLAY, SAGE_exp, K²-Agent)
│       │   └── Experience Revision
│       │             (BEPA, Comp.RL, INSPO, PEARL, AgentEvolver, DGO)
│       └── Optimized Experience Base
│           ├── Experience-derived Utility
│           │         (D2Skill, SLEA-RL, Skill-SD)
│           └── Trainable Experience Module
│                     (Trainable Graph Memory, UMEM)
│
├── §5 Domain-specific Hints
│   ├── Technical Domains
│   │         (Kevin, SGS, C2F-Thinker)
│   ├── Interactive Agent Domains
│   │         (COS-PLAY, WebGen-Agent, UI-S1)
│   └── Business & Personal-service Domains
│             (TaoSR-AGRL, CRMWeaver, VeriRole, PEARL)
│
└── §6 Discussion, Challenges and Future Directions
    ├── §6.1 Scope and Boundaries
    ├── §6.2 Cross-level Analysis of Hints
    └── §6.3 Challenges and Future Directionsurce, utilization mechanism, inference-time hint use)
```
