from pettingzoo.mpe import simple_tag_v3
env = simple_tag_v3.parallel_env(N=4, max_cycles=25, continuous_actions=False, render_mode="human")

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy

    env.step(action)
env.close()