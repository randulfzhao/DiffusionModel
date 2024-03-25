# 1. Research Overview

1. Idea: adopt classifier-free probalistic diffusion model to predict state for offline reinforcement learning based decision-making algorithm
2. keywords
	1. conditional generative model, RL, decision making
3. Research Question
	1. if we can perform dynamic programming to stich together sub-optimal trajectories to obtain an optimal trajectory without relying on value estimation
4. Contributions
	1. illustrating **conditional generative model** as effective tool in **offline** decision making
	2. use classifier-free guidance with low-temperature sampling rather than dynamic programming to get return-maximizing trajectories
	3. leverage framework of conditional generative model to combine constraints and compose skills during inference flexibly

# 2. Feedbacks

## 2.1. Questions

1. about the preliminaries:
	1. how is "low-temperature" =="low"== (appendix c, while ==not understood==)
	2. can TD methods be adopted for time-series generation? maybe we can set up the long-term generation problem in combined framework of TD and diffusion
2. about the framework
	1. how the diffusion works with the states? (根据前面C个state diffusion生成下一个state的信息，然后根据reward选择action function, 然后执行？)

## 2.2. To-do

1. LR
	1. diffusion model + Planning: 
		1. (experiment & architecture) U-Net: Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In International Conference on Machine Learning, 2022.
		2. (model) perturbed noise: Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. arXiv preprint arXiv:2206.01714, 2022.
	2. guided diffusion:
		1. (classifier free) Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
2. experiment
	1. 部署一个toy diffusion model for image generation / RL
	2. 本地运行 & 测试CPDD model

# 3. Premilinaries: Definitions and notations

## 3.1. Reinforcement Learning

1. Following MDP RL framework, the sequential decision-making problem is described as $\langle \rho_0, \mathcal S, \mathcal A, \mathcal T, \mathcal R, \gamma \rangle$, i.e., $\langle \text{initial state distribution, state spaces, action spaces, } \text{transition function, reward function, discount factor} \rangle$
2. stochastic polisy $\pi: \mathcal S \to \Delta_\mathcal A$ generates a sequence of state-action-reward transitions / trajectory $\tau:=(s_k, a_k, r_k)_{k\geq 0}$ with probability $p_\pi(\tau)$ and return $R(\tau):=\sum_{k\geq 0} \gamma^k r_k$ 
3. standard objective: $\pi^* = \text{arg} \min_\pi \mathbb E_{\tau \sim p_\pi} [R(\tau)]$
4. temporal difference learning (TD): estimate parameterized Q-function using TD loss: $$\mathcal L_{TD}(\theta) := \mathbb E_{(s,a,r,s') \in \mathcal D}[(r+\gamma \max_{a'\in \mathcal A} Q_\theta (s', a') - Q_\theta(s,a))^2]$$
5. Offline RL: utilize a return-maximizing policy from a fixed dataset of transitions collected by an unknown behavior policy $\mu$

## 3.2. Diffusion model
1. diffusion probabilistic models
	1. forward noising process (predefined): $q(x_{k+1}|x_k):=\mathcal N(x_{k+1};\sqrt \alpha_k x_k, (1-\alpha_k)\mathbf I)$ 
	2. reverse process (trainable): $p_\theta (x_{k+1}|x_k):= \mathcal N(x_{k-1}|\mu_\theta (x_k, k), \Sigma_k)$ with Gaussian distribution $\mathcal N(\mu, \Sigma)$
2. guided diffusion: model conditional data distribution $q(x|y)$ with score matching
	1. classifier-guided: train additional classifier $p_\Phi(y|x_k)$ on noisy data
	2. classifier-free: modify the original traning setup to learn both conditional $\epsilon_\theta(x_k, y, k)$ and unconditional $\epsilon_\theta (x_k, k)$ for noise. (where $y$ is a dummy value $\emptyset$)

# 4. Proposed pipeline

sequential decision making as standard problem of conditional generative model $$\max_\theta \mathbb E_{\tau \sim \mathcal D}[\log p_\theta(x_0(\tau)|y(\tau))]$$
## 4.1. setup

1. How to view the problem as diffusion process?
	1. model action as images is unpractical
		1. actions are often discrete
		2. sequence over actions change frequently and is non-smooth
	2. choose to ==diffuse over states==, where $x_k(\tau):=(s_t, s_{t+1}, \cdots, s_{t+H-1})_k$, where $k$ denotes the timestep in forward process, $t$ denotes the time at which a state was visited in trajectory $\tau$
	3. inverse dynamic: generate action using 2 consecutive states $a_t := f_\Phi(s_t, s_{t+1})$
2. why we plan with classifier-free:
	1. tradational: 
		1. train classifier $p_\Phi(y(\tau)|x_k(\tau))$ to predict $y(\tau)$ from noisy trajectory $x_k(\tau)$
			1. require separate, complex dynamic programming procedure
		2. directly train conditional diffusion model conditioned on returns $y(\tau)$
			1. polluted by sub-optimal trajectories
	2. classifier-free guidance with low-temperature sampling: extract high-likelihood trajector
		1. perturbed noise
3. architecture: 
	1. temporal U-Net architecture, to treat sequence of states $x_k(\tau)$ as image, where 
		1. height represents dimension of single state, 
		2. width represent length of trajectory
	2. encoding: conditional information $y(\tau)$ is encoded as one-hot vector or scalar, projected on latent variable $z \in \mathbb R^h$ with MLP
	3. low-temperature sampling for ablation study

## 4.2. algorithm
![[conditional_planning_dicision_diffuser_algorithm.jpg]]
1. observe state in environment
2. sample states later with diffusion process conditioned on $y$ and history of last $C$ states observed
3. identify the action shoule be taken to reach most immediate predicted state