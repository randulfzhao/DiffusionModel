# 1. aim

transfer the online fine-tuning of diffusion model from discrete into continuous policy space $\Rightarrow$ modify the ==diffusion generation== and ==fine-tuning== stage
- formulate the diffusion model (denoising process) as a RL problem

# 2. previous work

## 2.1. framework components

- SEIKO: online fine-tuning diffusion model by iteratively updating diffusion model and reward at the same time 
- continuous model-based RL: continuous policy space, adopt MSS to choose the strategy execution time.

## 2.2. related works

- online learning framework
	- [(2023, TMLR) RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment | RAFT](https://openreview.net/forum?id=m7p5O7zblY): general online learning approach for generative models. The fine-tuning should be in form of $$\max _w\left[\mathbb{E}_{x \sim D, y \sim p_g(\cdot \mid w, x)} r(x, y)-\beta Q(w)\right]$$ with loss $Q(w)$. $Q(\cdot)$ can often be KL divergence
 
- continuous time diffusion model framework
	- [(2021, ICLR) SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS](https://arxiv.org/abs/2011.13456): model score-based diffusion model (cts)
	- [(2023, Nips) Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://openreview.net/forum?id=HPuSIXJaa9): plausible direction: formulate the reward model as a DPO
- diffusion model in form of RL
	- [(2023, ICLR) Diffusion policies as an expressive policy class for offline rl](https://arxiv.org/abs/2208.06193): use Q-learning guidance into learning of diffusion model in training stage (not that related)
- RL for fine-tuning Text2Image diffusion models:
	- [(2023, NeurIPS) DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models|DPOK](https://arxiv.org/abs/2305.16381): use a discrete time MDP as the denoising process: optimiziing the policy is equivalent to the denoising process
	- [(2024, ICLR) Training Diffusion Models with Reinforcement Learning|ddpo](https://arxiv.org/abs/2305.13301): training denoising diffusion models to directly optimize a variety of reward functions (similar to DPOK. re-enter for design of reward)
- continuous-time RL by score matching:
	- [(ongoing work) Scores as Actions: a framework of fine-tuning diffusion models by continuous-time reinforcement learning](https://arxiv.org/abs/2409.08400)
		- no correlation with our framework
- oracle implementation
	- [(2016, NeurIPS) Deep exploration via bootstrapped DQN](https://arxiv.org/abs/1602.04621): bootstrap in neural network as oracle (original as outputing the Q-value using NN), where:
		- use $K$ separate neural networks, each with own parameter. each network output own estimate of the target for different actions
		- at each time step, use masking technique to decide when the network should be updated
		- random network selection: for each new episode, randomly select a network to pick actions during the entire episode
		- use replay buffer to store past experiences encountered during the episodes. sample from the buffer to update the network
		- update the networks selected from the mask using opt techniques
	- [(2018, NeurIPS) Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models|PETS](https://arxiv.org/abs/1805.12114): propose ensemble algorithm (probabilistic ensembles with trajectory sampling (PETS) that combines uncertainty-aware deep network dynamics models with sampling-based uncertainty propagation), where:
		- 


# 3. Formulation

## 3.1. Overview

- goal: 
	- diffusion model: continuous dynamic with discrete control. 
		- The state evolves continuously according to an SDE. 
		- The control actions are applied at selected discrete time points $\{ t_k \}$, determined by the measurement selection strategy (MSS)
	- fine-tuning

## 3.2. representing the fine-tuning of diffusion process as a continuous time MDP

We consider the fine-tuning of a text-to-image diffusion model within the framework of a continuous-time Markov Decision Process (MDP). Time is denoted in reverse, i.e., as time $t$ increases, the denoising process progresses, moving backward through the original diffusion process. 
- input:
	- pretrained naive diffusion model $f^{pre}$
	- text token $z$ as the instruction
- state space $\mathcal S$: the set of all possible noisy images at different levels of noise. The state $x(t)\in \mathbb R^n$ represents the image data corrupted by noise corresponding to that time.
- action space $\mathcal A$: the action $u(t, x(t), z)\in \mathbb R^n$ is a continuous control input that represents the denoising operation applied at time $t$. This control input is designed to improve image quality conditioned on the text token $z$.
- policy $\pi = \{(t_k,u_{t_k})\}_{k=0}^N$ defines both when to apply control actions based on MSS $\mu$ and the control strategy $\nu$, where
	- MSS $\mu$ determines the next control time $t_{k+1}$ based on the current state $x(t^+_k)$ and time $t_k$, i.e., $$t_{k+1} = \mu(x(t_k^+),t_k)$$
	- control strategy $\nu$ determines the control action $u_{t_k}$ at time $t_k$ based on the state $x(t_k^-)$, time $t_k$ and text token $z$, i.e., $$u_{t_k} = \nu(x(t_k^-),t_k,z)$$
- State Dynamics: we can express the state evolution over the entire time horizon $[0,T]$ as  $$x(T) = x(0) +\int_0^T p_{pre}(t,x(t))dt + \int_0^T \sigma(t)dw_t + \sum_{t_k\leq T} u_{t_k}$$ or equivalently, $$dx(t) = [p_{pre} (t,x(t)) + \sum_k \delta(t-t_k)u_{t_k} ] dt + \sigma(t) dw_t $$where
	- drift coefficient $p_{pre}(t,x(t)):[0,T]\times \mathcal X\to \mathcal X^d$ is the drift coefficient from the pretrained model $f_{pre}$
	- initial distribution / initial distribution  $x(0) \sim d_{pre}$
	- diffusion coefficient associated with Brownian motion $w_t$: $\sigma:[0:T]\to \mathbb R_{>0}$
	- $w_t$ is the standard Brownian motion
	- $\delta(t-t_k)$: 	Dirac delta function representing instantaneous control at $t_k$
- policy $\pi$ is learnt to optimize the reward function, deriving the denoising process to produce high-quality images. $\pi$ is a combination of 
	- Measurement Selection Strategy (MSS) ($\mu$) determines the next control time $t_{k+1}$ based on current state $x_{t_k}$ and time $t_k$
	- Control strategy $\nu$ determining the control action $u_{t_k}$ at time $t_k$, i.e., $$\pi = \{(t_k, u_{t_k})\}_{k=0}^N, ~where~t_{k+1} = \mu(x_{t_k},t_k)$$
- objective: accumulation of reward with KL regularization for the fine-tuning amount in the whole process, i.e., update the diffusion by $$\hat \theta = arg\max_\theta \mathbb E_{p(z)}[\mathbb E_{p_\theta (x_T|z)}[r(x_T,z)]] - \alpha \operatorname{KL}(p_\theta(x_T|z) \| p_{pre}(x_T|z))$$ for reward function $r(x_T,z)$ evaluating the generated image's quality and the KL term controlling the amount of fine-tuning to avoid overfitting following [1](https://arxiv.org/abs/2203.02155) and [2](https://arxiv.org/abs/2009.01325)

## 3.3. continuous-seiko: iteratively fine-tuning the pre-trained diffusion model under continuous setting

### 3.3.1. overview

- **Agent:** The diffusion model parameterized by $\thetaθ$.
- **Environment:** The process of denoising images over time.
- **State Space ($\mathcal{S}$):** The set of noisy images $x(t)$ at different noise levels.
- **Action Space ($\mathcal{A}$):** The denoising actions $u(t, x(t), z)$ applied to the images.
- **Policy ($\pi_\theta$​):** Defines how the agent chooses actions based on the current state and text token $z$.
- **Reward Function ($r(x_T, z)$):** Measures the quality of the final generated image conditioned on $z$.


### 3.3.2. algorithm

1.  Initialization
	1. initialize Parameter $\alpha, \{\beta_i\}\in \mathbb R^+$
	2. load ResNet18 (or some pretrained weak reward model) as initial reward model $\hat r^{(0)}$ , load some pretrained strong reward model as $r$
	3. Load `minimal-diffusion` model as pre-trained diffusion model $f^{pre}$ and $d^{pre}$: Set initial diffusion model: $f^{(0)} = f^{\text{pre}}$, set initial distribution: $d^{(0)} = d^{\text{pre}}$
	4. initialize policy model $\pi = \{(t_k, u_{t_k}\}_{k=0}^N$, with adaptive MSS $\mu$, selecting the control points $\{t_k\}$ and the control action $u_{t_k}$ (i.e., $t_{k+1} = \mu(x_{t_k},t_k)$) 
2. iterative process: For each iteration $i$ from $1:K$
	1. generative sample
		1. draw a batch of text instructions $\{z_i\}$ from data distribution $p(z)$
		2. for each $z_i$, sample $x^{(i)}_0$ from the curent noisy distribution $d_{i-1}$
		3. run the diffusion process to get the denoised sample $x_T^{(i)}$ from SDE: $$dx_t = [f^{(i-1)}(t, x_t)+\sum_k \delta(t-t_k)u_{t_k}]  dt + \sigma(t) dW_t$$
		4. classify based on the ground truth model, $y^{(i)} = r(x^{(i)}_T) + \epsilon$
	2. update dataset $\mathcal{D}^{(i)} = \mathcal{D}^{(i-1)} \cup \{(x^{(i)}_T, y^{(i)})\}$
	3. train reward $\hat r^{(i)}$ and the uncertainty oracle $\hat g^{(i)}(x)$ using dataset $\mathcal D^{(i)}$
		1.  update $\hat r$ as a regularized empirical risk minimization$$\hat{r}^{(i)}(\cdot)=\underset{r \in \mathcal{F}}{\operatorname{argmin}} \sum_{(x, y) \sim \mathcal{D}^{(i)}}\{r(x)-y\}^2+\|r\|_{\mathcal{F}}$$where with $\mathcal{F}$ as a hypothesis class such that $\mathcal{F} \subset[\mathcal{X} \rightarrow \mathbb{R}]$ and $\|\cdot\|_{\mathcal{F}}$ is a certain norm to define a regularizer.
		2. update $\hat g^{(i)}$, where with probability $1-\delta$, $x \in \mathcal X_{pre}, |\hat r^{(i)}(x) - r(x)|\leq \hat g^{(i)}(x)$. Use bootstrapped neural network for $\mathcal F$
	4. Optimize Diffusion Model: solve the following optimization problem $$\theta^{(i)}, d^{(i)} = arg\max_{\theta:[0,T]\times \mathcal X \to \mathcal X, d^{(i)}\in\Delta(\mathcal X)} \mathbb E_{p(z)}[\mathbb E_{p_\theta (x_T|z)}[(\hat r^{(i)} + \hat g^{(i)})(x_T,z)]] - \alpha \operatorname{KL}(p_\theta(x_T|z) \| p_{pre}(x_T|z)) - \beta_{i-1}\operatorname{KL}(p_{\theta^{(i)}}(x_T|z)\| p_{\theta^{(i-1)}}(x_T|z))$$




### 3.3.3. demo

following the settings under SEIKO and continuous RL, we want to implement a Toy Model for Iteratively Updating Minimal-Diffusion

1.  Initialization
	1. initialize Parameter $\alpha, \{\beta_i\}\in \mathbb R^+$
	2. load pretrained weak ResNet18 (accuracy: 36.86%) as initial reward model $\hat r^{(0)}$ , load pretrained strong reward model ResNet18 (acc: 95.36) as $r$
	3. Load `minimal-diffusion` model as the pre-trained diffusion model $f^{pre}_\theta$, containing the initial diffusion network $\theta^{pre}$ and the initial distribution $d^{pre}$. Set the initial diffusion model and distribution: 
		- $f^{(0)}_\theta = f^{\text{pre}}_\theta$ 
		- $d^{(0)} = d^{pre}$
	4. Initialize the uncertainty oracle $\hat g^{(0)}$
		1. create an ensemble of $M$ neural networks
		2. for each model $m=1$ to $M$, 
			1. generate a bootstrapped dataset $\mathcal D_m^{0}$ by sampling with replacement from an initial dataset $\mathcal D^{(0)}$ 
			2. use the loaded initial reward model $\hat r^{(0)}$ to predict rewards
		3. define uncertainty estimate $\hat g^{(0)}(x)$: for any input $x$, compute the standard deviation of ensemble predictions as $$\begin{aligned} &\hat{g}^{(0)}(x)=\sqrt{\frac{1}{M-1} \sum_{m=1}^M\left(\hat{r}_m^{(0)}(x)-\bar{r}^{(0)}(x)\right)^2}\end{aligned}$$where $\bar{r}^{(0)}(x)=\frac{1}{M} \sum_{m=1}^M \hat{r}_m^{(0)}(x)$ 
		4. confidence guarantee: with probability $1-\delta$, for $x\in \mathcal X_{pre}$, $$|\hat r^{(0)}(x) - r(x) \leq \hat g^{(0)}(x)$$
	5. Initialize the mapping of CIFAR-10's class names.
2. iterative fine tuning: For each iteration $i$ from $1:K$
	1. generate new samples
		1. draw a batch of text instructions $\{z_i\}$ from data distribution $p(z)$
		2. For each $z_i$, sample $x^{(i)}_0$ from distribution $d^{(i)}$ using the procedures below:
			1. Run the denoising diffusion process to obtain the hallucinated trajectory $\hat{x}_t^{(i)}$ by solving: $$ dx_t = f^{(i-1)}(t, x_t) dt + \sigma(t) dW_t $$
			2. Measure state-action pair $z^{(i)}(t)$: at uniform time intervals $\Delta t$, record the hallucinated trajectory's states $\hat{x}^{(i)}_t$ and the corresponding control strategy $\pi^{(i-1)}(\hat{x}^{(i)}_t)$
			3. Compute uncertainty using the uncertainty model $\hat{g}^{(i-1)}$ from the previous round (containing $M$ ensemble nn), get $\sigma^{(i-1)}(t) = \hat g^{(i-1)}$
			4. identify high-uncertainty points: select time points $t_k$ where uncertainty exceeds a threshold or pick the top $k$ highest uncertainties
			5. compute control actions $$u_{t_k} = \pi^{(i-1)}(\hat x_{t_k}^{(i)})$$
			6. Run the diffusion process with control to obtain the denoised sample $x^{(i)}_T$: $$ dx_t = \left[ f^{(i-1)}(t, x_t) + \sum_k \delta(t - t_k) u_{t_k} \right] dt + \sigma(t) dW_t $$where $\delta(\cdot)$ is the Dirac function, representing instantaneous control at time point $t_k$.
		3. compute reward based on the ground truth model: $$y^{(i)} = r(x^{(i)}_T) + \epsilon$$ where $\epsilon$ is a small noise term
	2. update dataset $\mathcal{D}^{(i)} = \mathcal{D}^{(i-1)} \cup \{(x^{(i)}_T, y^{(i)})\}$
	3. train reward $\hat r^{(i)}$ and the uncertainty oracle $\hat g^{(i)}(x)$ using dataset $\mathcal D^{(i)}$ using bootstrapping Neural Network, with confidence guarantee: with probability $1-\delta$, $x \in \mathcal X_{pre}, |\hat r^{(i)}(x) - r(x)|\leq \hat g^{(i)}(x)$
		1. create $M$ bootstrapped dataset $\{\mathcal D_m^{(i)}\}$ from $\mathcal D^{(i)}$
		2. for each $m=1$ to $M$, train $\hat r_m^{(i)}$ on $\mathcal D_m^{(i)}$, with $\hat r$ as a regularized empirical risk minimization$$\hat{r}^{(i)}_m(\cdot)=\underset{r \in \mathcal{F}}{\operatorname{argmin}} \sum_{(x, y) \sim \mathcal{D}_m^{(i)}}\{r(x)-y\}^2+\lambda \|r\|_{\mathcal{F}}$$where $\mathcal{F}$ is a hypothesis class such that $\mathcal{F} \subset[\mathcal{X} \rightarrow \mathbb{R}]$ and $\|\cdot\|_{\mathcal{F}}$ is the $\ell_2$ norm
		3. update $\hat r^{(i)} = \frac{1}{M} \sum_{m=1}^M \hat r_m^{(i)}(\cdot)$
		4. compute the uncertainty estimate: $$\forall x,~\hat{g}^{(i)}(x)=\sqrt{\frac{1}{M-1} \sum_{m=1}^M\left(\hat{r}_m^{(i)}(x)-\hat{r}^{(i)}(x)\right)^2}$$
	4. Optimize Diffusion Model: solve the following optimization problem $$ f^{(i)}_\theta, d^{(i)} = \arg\max_{f:[0,T] \times \mathcal{X} \to \mathcal{X}, d^{(i)} \in \Delta(\mathcal{X})} \mathbb{E}_{p(z)} \left[ \mathbb{E}_{p_\theta (x_T | z)} \left[ (\hat{r}^{(i)} + \hat{g}^{(i)})(x_T, z) \right] \right] - \alpha \operatorname{KL}(p_\theta(x_T | z) \| p_{pre}(x_T | z)) - \beta_{i-1} \operatorname{KL}(p_{\theta^{(i)}}(x_T | z) \| p_{\theta^{(i-1)}}(x_T | z)) $$