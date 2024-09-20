# 1. aim

transfer the online fine-tuning of diffusion model from discrete into continuous policy space $\Rightarrow$ modify the ==diffusion generation== and ==fine-tuning== stage
- formulate the diffusion model (denoising process) as a RL problem

# 2. previous work

## 2.1. framework components

- SEIKO: online fine-tuning diffusion model by iteratively updating diffusion model and reward at the same time 
- continuous model-based RL: continuous policy space, adopt MSS to choose the strategy execution time.

## 2.2. related works

- diffusion model in form of RL
	- [(2023, ICLR) Diffusion policies as an expressive policy class for offline rl](https://arxiv.org/abs/2208.06193): use Q-learning guidance into learning of diffusion model in training stage (not that related)
- RL for fine-tuning Text2Image diffusion models:
	- [(2023, NeurIPS) DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models|DPOK](https://arxiv.org/abs/2305.16381): use a discrete time MDP as the denoising process: optimiziing the policy is equivalent to the denoising process
	- [(2024, ICLR) Training Diffusion Models with Reinforcement Learning|ddpo](https://arxiv.org/abs/2305.13301): training denoising diffusion models to directly optimize a variety of reward functions (similar to DPOK. re-enter for design of reward)
- continuous-time RL by score matching:
	- [(ongoing work) Scores as Actions: a framework of fine-tuning diffusion models by continuous-time reinforcement learning](https://arxiv.org/abs/2409.08400)
		- no correlation with our framework


# 3. Formulation

formulate denoising of DDPM as a continuous time MDP, to make it equivalent as to fine-tuning the underlying DDPM. the diffusion model here is a text2image diffusion model, with input token (text) $z$
- denote the time reversely, i.e., as time $t$ increase, the denoising process go forward, backing to *previous time step* in original set of notations
- input:
	- pretrained naive diffusion model $f^{pre}$
- state space $S$: the set of all possible noisy images at different levels of noise
- state dynamic as SDE: $dx(t) = f^{pre}(t,x(t)) dt + \sigma(t) dw_t$, where 
	- drift coefficient $f^{pre}:[0,T]\times \mathcal X\to \mathcal X^d$ controls the deterministic part of the state evolution
	- initial distribution $x(0) \sim \nu^{pre} = \mathcal N(0,I)$
	- diffusion coefficient associated with Brownian motion $w_t$: $\sigma:[0:T]\to \mathbb R_{>0}$
- ==action representation==: the action should be modifying the drift coefficient to influence the denoising process, i.e., $a(t,x(t),dt) = x(t+dt)$, where $dt$ should be determined by the measurement selection strategy *MSS*
- policy $\pi(t,x(t))$ from continuous control space determines the action $a(t,x(t),dt)$ at each time $t$ given the current state $x(t)$. policy is learnt to optimize the reward function, deriving the denoising process to produce high-quality images
- objective: accumulation of reward in the whole process, i.e., update the diffusion by $$\max_\theta \mathbb E_{p(z)}[\int_0^T R(x(t),\pi(t,x(t)))dt] - KL(\cdots)$$ for reward function at time $t$ being $R(x(t),\pi(t,x(t)))$
	- $R(x(t),\pi(t,x(t))) = r(x_T,z)$ at the final step, using some classifier / judge to evaluate the quality of generated figure w.r.t. token $z$, otherwise $0$
		- ==is this equivalent to the denoising process of a diffusion model?==
		- ==evaluation function should contain the information of policy?==
- 