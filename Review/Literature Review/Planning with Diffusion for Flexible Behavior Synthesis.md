# 1. Research Overview

1. Idea: diffusion probablistic model that plan by iteratively denoising trajectories. Diffuser predicts all timesteps of a plan simultaneously
2. keywords
3. Research Question
4. Contributions: propose the model **diffuser** that 
	1. long-horizon scalability & temporal compositionality: capable to generate rajectories rather than single-step, iteratively improving local consistency
	2. task compositionality: reward function provide auxiliary gradients to be used while sampling a plan
	3. no-greedy planning: training procedure can avoid long-horizon, sparse-reward problems

# 2. Feedbacks

## 2.1. Questions

1. preliminaries:
	1. what is "auto-regressive" 
	2. what is "U-Nets"
	3. what is Dirac Delta

## 2.2. To-do

1. Literature review: 
	1. basic diffusion probablistic models:
		1. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, 2015.
		2. Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, 2020.
		3. 

# 3. Premilinaries: Definitions and notations

## 3.1. trajectory optimization

1. objective: we want to find a sequence of actions $\mathbf a_{0:T}^*$ that maximize the objective $\mathcal J$ factorized over per-timestep rewards $r(s_t, a_t)$ in discrete-time dynamics ($s_{t+1} = f(s_t, a_t)$ at state $s_t$ given action $a_t$) given the planning horizon $T$. i.e., $$\mathbf a_{0:T}^* = \arg \min_{\mathbf a_{0:T}} \mathcal J(s_0, a_{0:T}) = \arg\min_{\mathbf a_{0:T}} \sum_{t=0}^T r(s_t, a_t)$$ we use $\tau = (s_0, a_0, s_1, a_1, \cdots, s_T, a_T)$ to refer to a trajectory of interleaved states and actions, with $\mathcal J(\tau)$ denoting objective value of the trajectory
2. obstacle: require knowledge of environment dynamics $f$
	1. train an approximate dynamics model, plugging into conventional planning routine
		1. cannot suit planning algorithm with ground truth in mind
	2. proposed model: ==subsume planning process into generative modeling framework as much as possiible==, then planning $\equiv$ diffusion generating ^f44dcf
		1. $p_\theta(\tau) \propto p_\theta(\tau)h(\tau)$  ^c9e879

## 3.2. diffusion probablistic models

1. data-generating process as iterative denoising procedure $p_\theta(\tau^{i-1}|\tau^i)$, where the forward diffusion process is denoted as $q(\tau^i|\tau^{i-1})$
	1. with $p(\tau^N)$ denoted as standard Gaussian prior and $\tau^0$ as noiseless data, induced data distribution: $$p_\theta(\tau^0) = \int p(\tau^N) \prod_{i=1}^Np_\theta(\tau^{i-1}|\tau^i)d\tau^{1:N}$$ 
	2. $\theta$ are optimized by minimizing variational bound on negative log likelihood of reverse process: $\theta^* = \arg \min_\theta -\mathbb E_{\tau^0}[\log p_\theta(\tau^0)]$ 
	3. the ==reverse== process is typically parameterized as Gaussian sith fixed timestep-dependent covariances: $p_\theta(\tau^{i-1}|\tau^i) = \mathcal N(\tau^{i-1}|\mu_\theta(\tau^i, i), \Sigma^i)$
	4. the forward process is typically ==prespecified==


# 4. Proposed pipeline

## 4.1. Overview: planning with diffusion

1. obstacle:
	1. trajectory optimization: require knowledge of environment, and learnt models are often poorly suited to the types of planning algorithms designed with ground-truth model 
2. Solution: tighter couple between modeling and planning, making it nearly identical to sampling
	1. diffusion model of trajectories: $p_\theta(\tau)$
	2. denoising process: $\tilde p_\theta(\tau) \propto p_\theta(\tau)h(\tau)$, with $h(\tau)$ containing information of prior evidence / desired outcomes / general functions to optmize

## 4.2. setup

### 4.2.1. Diffusion

1. temporal ordering
	1. goal-conditioned inference: $p(s_1|s_0, s_T)$: the next state $s_1$ depends on a future state and a prior one
	2. **Diffuser** predicts all timesteps of a plan concurrently
2. temporal locality
	1. relaxed form of temporal locality: receptive field of a given prediction only consist of nearby timesteps (both in the past and future) $\to$ each step of denoising process can only make predictions based on local consistency of the trajectory
3. trajectory representation
	1. states and actions in a trajectory are predicted ==jointly==, in form of
	   ![[diffuser_state_action_prediction.png]]

### 4.2.2. reinforcement learning as guided sampling

1. reward: $\mathcal O_t$ is a binary random variable denoting the optimality of timestep $t$ of a trajectory
	1. $p(\mathcal O_t = 1) = \exp(r(s_t, a_t))$ 
	2. sample by setting $h(\tau) = p(\mathcal O_{1:T}|\tau)$ 
2. when $p(\mathcal O_{1:T}|\tau^i)$ is sufficiently smooth, the reverse diffusion process transitions can be approximated as Gaussian: $$p_\theta(\tau^{i-1}|\tau^i, \mathcal O_{1:T}) \approx \mathcal N(\tau^{i-1}; \mu + \Sigma g, \Sigma$$ with $\mu, \Sigma$ being parameters of original reverse process transition $p_\theta(\tau^{i-1}| \tau^i)$ and $$\begin{align} g &= \nabla_\tau \log p(\mathcal O_{1:T} | \tau) |_{\tau = \mu} \\ &= \Sigma_{t=0}^T r(s_t, a_t)|_{(s_t, a_t) = \mu_t} \\ &= \nabla \mathcal J(\mu) \end{align}  $$
	1. **transfer classifier-guided sampling** to **reinforcement learning problem**

## 4.3. algorithm

### 4.3.1. Architecture of diffuser: 
- entire trajectory should be non-autoregressively
- each step of denoising should be temporally local
- trajectory representation should allow for equivariance along planning horizon rather than state and action features
  ![[diffuser_guided.png]]

### 4.3.2. training

1. objective: with learned gradient $\epsilon_\theta(\tau^i, i)$, mean $\mu_\theta$, the objective of simplified $\epsilon$-model is $$\mathcal L(\theta) = \mathbb E_{i, \epsilon, \tau^0}[\|\epsilon - \epsilon_\theta (\tau^i, i) \|^2]$$ where $i\sim \mathcal U\{1, 2, \cdots, N\}$ is diffusion timestep, $\epsilon \sim \mathcal N(0, I)$ is noise target, $\tau^i$ is trajectory $\tau^0$ corrupted with noise $\epsilon$, reverse process covariance $\Sigma^i$ follows cosine scheme
2. problem solving: treat RL as ==conditional sampling==, with
	1.  $\mathcal O_t$: RV denoting optimality of timestep $t$ of a trajectory, where $p(\mathcal O_t = 1) = exp(r(s_t, \mathbf a_t))$ 
	2. setting $h(\tau) = p(\mathcal O_{1:T}|\tau)$ in [[Planning with Diffusion for Flexible Behavior Synthesis#^c9e879|original formula]]$\tilde p_\theta(\tau) = p(\tau | \mathcal O_{1:T} = 1) \propto p(\tau)p(\mathcal O_{1:T} = 1|\tau)$, where 
		1. reverse diffusion process transion can be approximated as Gaussian, $$p_\theta(\tau^{i-1}|\tau^i, \mathcal O_{1:T}) \approx \mathcal N(\tau^{i-1};\mu + \Sigma g, \Sigma)$$ with $\mu, \Sigma$ being parameters of original reverse process transition $p_\theta (\tau^{i-1}|\tau^i)$, $g$ approached through $$\begin{align} g &= \nabla_\tau \log p(\mathcal O_{1:T}|\tau)|_{\tau = p} \\ &= \sum_{t=0}^T \nabla_{\mathbf s_t, \mathbf a_t} r(\mathbf s_t, \mathbf a_t)|_{(\mathbf s_t, \mathbf a_t) = \mu_t} \\ &= \nabla \mathcal J(\mu)\end{align}$$ used to modify the mean $\mu$ and the reverse process
3. pertubation: require a Dirac Delta for observed values and constant. if $\mathbf c_t$ is state constraint at timestep $t$, then![[plan_with_diffusion_perturbation.png]]

## 4.4. experiment

### 4.4.1. test target

1. ability to plan over long horizons without manual reward shaping
2. ability to generalize to new configurations of unseen goals
3. ability to recover controller for heterogenerous data 

### 4.4.2. baselines

1. long horizon planning: Maze2D
2. test-time: 
	1. unconditional stacking: building a block tower as tall as possible. sampling from unpertubed denoising process $p_\theta(\tau)$ to emulate PDDLStream controller
	2. conditional stacking: construct a block tower with specified order of blocks
	3. rearrangement: match a set of reference blocks' location 