import argparse
import tensorflow as tf
import gym
import numpy as np
from ReplayBuffer import PrioritizedReplay, UniformReplay
from model import Model

hyperparams = {
    'gamma': 0.99,
    'copy_step': 25,
    'max_experiences': 10000,
    'min_experiences': 1000,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'environment': 'Acrobot-v1',
    'network': [128],
    'episodes': 100,
    'epsilon': 0.99,
    'min_epsilon': 0.1,
    'decay': 0.9,
}


def start_training(is_prioritized, is_double):
    if is_prioritized:
        prio = "Prio"
    else:
        prio = ""

    if is_double:
        ddqn = "double"
    else:
        ddqn = "single"

    env = gym.make(hyperparams['environment'])
    state_spec = len(env.observation_space.sample())
    action_spec = env.action_space.n
    log_name = 'final' + prio + ddqn
    log_dir = 'logs/acrobot/' + log_name
    log_writer = tf.summary.create_file_writer(log_dir)

    epsilon = hyperparams['epsilon']
    buffer = PrioritizedReplay(hyperparams['max_experiences']) if is_prioritized else UniformReplay(
        hyperparams['max_experiences'])

    agent = Model(hyperparams['network'], state_spec, action_spec, buffer, hyperparams['learning_rate'], is_prioritized,
                  is_double)

    time = 0
    total_rewards = np.empty(hyperparams['episodesf'])
    for episode in range(hyperparams['episodes']):
        episode_reward = 0
        epsilon = max(hyperparams['min_epsilon'], epsilon * hyperparams['decay'])
        # train loop

        # if i don't have enough samples i will not call train
        done = False
        state = env.reset()
        while not done:

            action = agent.play_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            buffer.add((state, action, reward, next_state, done))
            state = next_state

            if len(buffer.experiences) > hyperparams['min_experiences']:
                agent.train(hyperparams['gamma'], hyperparams['batch_size'])

            time += 1
            if time % hyperparams['copy_step'] == 0 and is_double:
                agent.update_target()
                time = 0

        total_rewards[episode] = episode_reward
        avg_rewards = total_rewards[max(0, episode - 100):(episode + 1)].mean()

        env.reset()
        if episode % 100 == 0:
            print(avg_rewards)

        with log_writer.as_default():
            tf.summary.scalar('episode reward', episode_reward, step=episode)
            tf.summary.scalar('avg for 100 episodes', avg_rewards, step=episode)

    env.close()


def test_model(model):
    env = gym.make(hyperparams['environment'])
    state_spec = len(env.observation_space.sample())
    action_spec = env.action_space.n
    buffer = None
    is_prioritized = False
    is_double = False
    # Create the network
    agent = Model(hyperparams['network'], state_spec, action_spec, buffer, hyperparams['learning_rate'], is_prioritized,
                  is_double)
    # Load weights from file
    agent.training_network.load_weights(model)
    obs = env.reset()
    env.render()
    for i in range(20):
        rewards = []
        while True:
            action = agent.play_action(np.atleast_2d(obs)[0], 0.1)
            obs, reward, done, _ = env.step(action)
            env.render()
            rewards.append(reward)
            if done:
                print("Gathered {} reward".format(np.sum(rewards)))
                env.reset()
                break

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help="One between {train|test}", choices=['train', 'test'], type=str, required=True,
                        dest='mode')
    parser.add_argument('--per', help="Use this if you want experience replay", action="store_true", dest='per',
                        default=False)
    parser.add_argument('--model', help="the model you want to test", type=str, dest='model')
    parser.add_argument("--double", help="Use double DQN", action="store_true", dest='double')
    args = parser.parse_args()
    if args.mode == 'train':
        print('TRAIN')
        print("PER", args.per)
        print("DOUBLE", args.double)
        start_training(args.per, args.double)
    elif args.mode == 'test':
        print('test')
        test_model(args.model)
