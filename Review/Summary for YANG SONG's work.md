# 1. RQ:
1. suppose we have collected some data, can we leverage those existing data to simulate the data generating process? 
	1. can we build a data simulator?

# 2. how to build the diffusion model

1. framework of probability distribution estimation![[estimating_probability_distribution.png]]
2. challanging:
	1. data distribution is complex for high dimensional data. hard to find the model fitting into the data distribution
		1. Gaussian
		2. DNN: $\mathbf x$(==original data==) $\to e^{f_\theta (\mathbf x)}$ (==make the value positive==) $\to \frac{e^{f_\theta (\mathbf x)}}{Z_\theta}$ (==normalize using constant $Z_\theta$==) $\to p_\theta(\mathbf x)$ (==normalized value==), where the constant is always set to be
			1. $Z_\theta = \int e^{f_\theta(\mathbf x)}d\mathbf x$ 
			2. $Z_\mu = \frac{1}{(2\pi)^{d/2}}$ for Gaussian Distribution
			3. where the constant is always intractable
			4. approach:
				1. approximate the normalizing constant (inaccurate probability evaluation)
				2. using restricted nn models (limited flexibility to restricted nn)
				3. GANs (cannot evaluate probabilities)

# 3. proposed framework:$f_\theta$

## 3.1. using score functions
1. benefits
	1. bypass the normalizing constant (computed easily using automatic differentiation / back propogation)
	2. improved generation quality
	3. accurate probability evaluation
2. details
	1. given: $\{\mathbf x_1, \mathbf x_2, \cdots, \mathbf x_N\} \overset{i.i.d.}{\sim} p_{data}(\mathbf x)$
	2. goal: $\nabla_\mathbf x \log p_{data}(\mathbf x)$
	3. score model: $s_\theta(\mathbf x): \mathbb R^d \to \mathbb R^d \approx \nabla_\mathbf x \log p_{data}(\mathbf x)$ 
	4. objective: how to compare 2 vector fields of scores? using fisher divergence: $$\begin{align}&\frac{1}{2} \mathbb E_{p_{data}(\mathbf x)}[\|\nabla_\mathbf x \log p_{data}(\mathbf x) - s_\theta(\mathbf x)\|_2^2]\\ \equiv & \mathbb E_{p_{data}(\mathbf x)} [\frac{1}{2} \|s_\theta(\mathbf x)\|_2^2 + trace(\nabla_\mathbf x s_\theta(\mathbf x))]  \end{align}$$ the first equation comes to the second equation using Gauss's theorem, integrating Jacobian of $s_\theta(\mathbf x)$ by parts. However, the trace is **not scalable**
		1. solution: using random projection. Sliced score matching
	5. generate high-quality samples from score model
		1. following the noisy scores (Langevin Dynamics): correct samples guaranteed.
			1. cannot work well with trouble in low-density regions. 
			2. level of noise will affect the result. to enhance, using Noise Conditional Score Model (**Annealed Langevin dynamics**, where $\sigma$ is another input of model). 
3. control the generation process: we want the backward model $p(x|y)$ given forward model $p(y|x)$ with control signal $y=label$ 
	1. Bayes' rule: $p(\mathbf x|y) = \frac{p(\mathbf x)p(y|\mathbf x)}{p(y)}$. we know the numerator, while we don't know the denominator
		1. Bayes' rule for score functions: $$\begin{align} \nabla_\mathbf x \log p(\mathbf x|y) &= \nabla_\mathbf x \log p(\mathbf x) + \nabla_\mathbf x \log p(y|\mathbf x) - \nabla_\mathbf x \log p(y) \text{(0)} \\ &=\nabla_\mathbf x \log p(\mathbf x) + \nabla_\mathbf x \log p(y|\mathbf x) \end{align}$$ where the first term is approximately unconditional score $s_\theta(\mathbf x)$, and the second term is gradient of logarithm of forward model. 
4. probability evaluation: 
	1. add noise based on stochastic processes![[pertubing_data_with_sde.jpg]]
	2. outcome: generate images from noise using reverse SDE
