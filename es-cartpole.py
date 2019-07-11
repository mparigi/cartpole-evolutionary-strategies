import gym
import numpy as np
import matplotlib.pyplot as plt


num_state_vars = 2

class Agent:
	def __init__(self, policy=(np.random.randn(num_state_vars), np.random.randn()), score=0, choices=2):
		self.score = score
		self.policy = policy
		self.choices = choices

	def act(self, obs):
		z = 0
		for w, o in zip(self.policy[0], obs):
			z += w*o
		z = z + self.policy[1]
		return self._map_to_choice(z, self.choices)

	def _map_to_choice(self, x, n):
		x = 1 / (1 + np.exp(-x)) # apply sigmoid
		x *= n # scale up to number of choices
		return int(np.floor(x)) # make discrete



def blur_policy(policy=(np.zeros(num_state_vars), 0)):
	return (policy[0] + np.random.randn(num_state_vars), policy[1] + np.random.randn())



env = gym.make('CartPole-v1')

lmbda = 12
mu = lmbda // 3 # mu must divide lambda

generations = 50
population = [Agent(choices=env.action_space.n)] * lmbda

best_of_each_gen = []

for generation in range(generations):
	# evaluate each individual
	for individual in population:
		observation = env.reset()
		for t in range(1000):
			env.render()
			action = individual.act(observation)
			observation, reward, done, info = env.step(action)
			if done:
				individual.score = t
				break

	# find the mu best individuals
	# they become the parents
	mu_best = sorted(population, key=lambda ind: ind.score)[-mu:]
	num_children = lmbda // mu
	new_population = []

	# find best individual
	best_of_each_gen.append(mu_best[-1].score)
	print(f"all scores of this generation: {[p.score for p in population]}")
	print(f"best of this generation: {best_of_each_gen[-1]}")
	print(f"average of this generation: {np.mean([p.score for p in population])}")
	print("")

	# each parent can produce children by blurring themselves
	for parent in mu_best:
		for _ in range(num_children):
			new_population.append(Agent(blur_policy(parent.policy), 0, choices=env.action_space.n))
	population = new_population




env.close()

plt.plot(best_of_each_gen)
plt.xlabel('generation')
plt.ylabel('best score in generation')
plt.show()
