#!/usr/bin/env python
import torch
from naf import NAF
from tensorboardX import pytorch_graph
from replay_memory import ReplayMemory, Transition
import numpy as np
import random
from ounoise import OUNoise
from Supervised_Learning import create_SL_model
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
import argparse
import os
import random
from keras.models import load_model
import pandas as pd


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



def fit_nash():
    suffix = 'Nash_{}_RC_{}_AttackMode_{}_RewardMode_{}'.format(args.NashMode, RC, args.AttackMode, args.RewardMode)
    # reward_file = open('reward' + suffix + '.txt', 'w')
    # attack_file = open('attacker_action' + suffix + '.txt', 'w')
    # weight_file = open('vehicle_weight' + suffix + '.txt', 'w')
    # distance_file = open('Distance' + suffix + '.txt', 'w')



#     reward_file.write("""
# Environment Initializing...
# The initial head car velocity is {}
# The initial safe distance is     {}
# The Nash Eq* Factor RC is        {}
# The Reward Calculation Mode is   {}
# The Attack Mode is               {}
# The Nash Mode is                 {}
# """.format(env.v_head, env.d0, RC, env.reward_mode, env.attack_mode, args.Nash))

    # reward_file.close()
    # attack_file.close()
    # weight_file.close()
    # distance_file.close()

    agent_vehicle = NAF(args.gamma, args.tau, args.hidden_size,
                        env.observation_space, env.vehicle_action_space, 'veh')
    agent_attacker = NAF(args.gamma, args.tau, args.hidden_size,
                         env.observation_space, env.attacker_action_space, 'att')
    try:
        agent_vehicle.load_model('models/vehicle_' + suffix)
        print('Load vehicle RL model successfully')

    except:
        print('No existed vehicle RL model')
    try:
        agent_attacker.load_model('models/attacker_' + suffix)
        print('Load attacker RL model successfully')

    except:
        print('No existed attacker RL model')
    try:
        policy_vehicle = load_model('models/vehicle_' + suffix + '.h5')
        print('Load vehicle SL model successfully')
    except:
        policy_vehicle = create_SL_model(env.observation_space, env.vehicle_action_space, 'vehicle')
    try:
        policy_attacker = load_model('models/attacker_' + suffix + '.h5')
        print('Load attacker SL model successfully')
    except:
        policy_attacker = create_SL_model(env.observation_space, env.attacker_action_space, 'attacker')
    print('*'*20, '\n\n\n')
    memory_vehicle = ReplayMemory(100000)
    memory_attacker = ReplayMemory(100000)

    memory_SL_vehicle = ReplayMemory(400000)
    memory_SL_attacker = ReplayMemory(400000)

    ounoise_vehicle = OUNoise(env.vehicle_action_space) if args.ou_noise else None
    ounoise_attacker = OUNoise(env.attacker_action_space) if args.ou_noise else None

    param_noise_vehicle = AdaptiveParamNoiseSpec(initial_stddev=0.05,
                                                 desired_action_stddev=args.noise_scale,
                                                 adaptation_coefficient=1.05) if args.param_noise else None
    param_noise_attacker = AdaptiveParamNoiseSpec(initial_stddev=0.05,
                                                  desired_action_stddev=args.noise_scale,
                                                  adaptation_coefficient=1.05) if args.param_noise else None
    try:
        res_data = pd.read_csv(suffix + '.csv', names=['Weight', 'Attack', 'Eva_distance'])
    except:
        res_data = pd.DataFrame(columns=['Weight', 'Attack', 'Eva_distance'])

    try:
        reward_data = pd.read_csv(suffix + '_reward_.csv', names=['Reward'])
    except:
        reward_data = pd.DataFrame(columns=['Reward'])

    rewards = []
    total_numsteps = 0
    for i_episode in range(args.num_episodes):
        if i_episode % 100 == 0 and i_episode != 0:
            print('Writing to CSV files...')
            reward_data.to_csv(suffix + '.csv', index=False)
            res_data.to_csv(suffix + '.csv', index=False)

        if args.NashMode == 0:
            ETA = 0
        elif args.NashMode in [1, 4]:
            ETA = 0.5
        elif args.NashMode == 2:
            ETA = 0.1 - i_episode/args.num_episodes * 0.1

        print('No.{} episode starts... ETA is {}'.format(i_episode, ETA))

        # reward_file = open('reward' + suffix + '.txt', 'a')
        # attack_file = open('attacker_action' + suffix + '.txt', 'a')
        # weight_file = open('vehicle_weight' + suffix + '.txt', 'a')
        # distance_file = open('Distance' + suffix + '.txt', 'a')

        state = env.reset()
        state_record = [np.array([state])]
        episode_steps = 0
        while len(state_record) < 20:
            a, b = env.random_action()
            s, _, _ = env.step(np.array([a]), np.zeros(4))
            state_record.append(s)
        env.step_number = 0
        if args.ou_noise:
            ounoise_vehicle.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                                      i_episode) / args.exploration_end + args.final_noise_scale
            ounoise_vehicle.reset()

            ounoise_attacker.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                                       i_episode) / args.exploration_end + args.final_noise_scale
            ounoise_attacker.reset()
        episode_reward = 0
        local_steps = 0
        while True:
            sigma = random.random()
            if sigma > ETA:
                action_vehicle = agent_vehicle.select_action(torch.Tensor(state_record[-20:]), ounoise_vehicle,
                                                             param_noise_vehicle)[:, -1, :]
            else:
                action_vehicle = torch.Tensor(
                    [policy_vehicle.predict(state_record[-1].reshape(-1, 4)) / policy_vehicle.predict(
                        state_record[-1].reshape(-1, 4)).sum()])[0]
                action_attacker = torch.Tensor(
                    [policy_attacker.predict(state_record[-1].reshape(-1, 4)) / policy_attacker.predict(
                        state_record[-1].reshape(-1, 4)).sum()])[0]

            # Nash_Mode = 4
            action_attacker = agent_attacker.select_action(torch.Tensor(state_record[-20:]), ounoise_attacker,
                                                           param_noise_attacker)[:, -1, :]

            action_vehicle = action_vehicle.numpy()[0]/(action_vehicle.numpy()[0].sum())
            action_attacker = action_attacker.numpy()[0]
            # attack_file.write(str(action_attacker) + '\n')
            # weight_file.write(str(action_vehicle) + '\n')
            next_state, reward, done = env.step(action_vehicle, action_attacker)
            res_data = res_data.append([{'Attack':action_attacker, 'Weight':action_vehicle, 'Eva_distance':env.d}])

            total_numsteps += 1
            episode_reward += reward

            state_record.append(next_state)
            local_steps += 1
            episode_steps += 1

            if sigma > ETA:
                memory_SL_vehicle.append(state_record[-1], action_vehicle)
                memory_SL_attacker.append(state_record[-1], action_attacker)

            action_vehicle = torch.Tensor(action_vehicle.reshape(1,4))
            action_attacker = torch.Tensor(action_attacker.reshape(1,4))

            mask = torch.Tensor([not done])

            prev_state = torch.Tensor(state_record[-20:]).transpose(0, 1)
            next_state = torch.Tensor([next_state])

            reward_vehicle = torch.Tensor([reward])
            reward_attacker = torch.Tensor([RC - reward])

            memory_vehicle.push(prev_state, torch.Tensor(action_vehicle), mask, next_state, reward_vehicle)
            memory_attacker.push(prev_state, torch.Tensor(action_attacker), mask, next_state, reward_attacker)

            if done:
                rewards.append(episode_reward)
                print('Episode {} ends, instant reward is {:.2f}'.format(i_episode, episode_reward))
                reward_data = reward_data.append([{'Reward': episode_reward}])
                    # reward_file.write('Episode {} ends, instant reward is {:.2f}\n'.format(i_episode, episode_reward))
                break

        if len(memory_vehicle) > args.batch_size:  # 开始训练
            for _ in range(args.updates_per_step):
                transitions_vehicle = memory_vehicle.sample(args.batch_size)
                batch_vehicle = Transition(*zip(*transitions_vehicle))

                transitions_attacker = memory_attacker.sample(args.batch_size)
                batch_attacker = Transition(*zip(*transitions_attacker))

                trans_veh = memory_SL_vehicle.sample(args.batch_size)
                trans_att = memory_SL_attacker.sample(args.batch_size)

                states_veh = []
                actions_veh = []
                states_att = []
                actions_att = []
                for sample in trans_veh:
                    state_veh, act_veh = sample
                    states_veh.append(state_veh)
                    actions_veh.append(act_veh)
                for sample in trans_att:
                    state_att, act_att = sample
                    states_att.append(state_att)
                    actions_att.append(act_att)

                states_veh = np.reshape(states_veh, (-1, env.observation_space))
                states_att = np.reshape(states_att, (-1, env.observation_space))
                actions_veh = np.reshape(actions_veh, (-1, env.vehicle_action_space))
                actions_att = np.reshape(actions_att, (-1, env.attacker_action_space))

                policy_vehicle.fit(states_veh, actions_veh, verbose=False)
                policy_attacker.fit(states_att, actions_att, verbose=False)
                agent_vehicle.update_parameters(batch_vehicle)
                agent_attacker.update_parameters(batch_attacker)

                # writer.add_scalar('loss/value', value_loss, updates)
                # writer.add_scalar('loss/policy', policy_loss, updates)

        # if i_episode % 10 == 0:
        #     # distance_file.write('{} episode starts, recording distance...\n'.format(i_episode))
        #     state = env.reset()
        #     state_record = [np.array([state])]
        #     evaluate_reward = 0
        #     while len(state_record) < 20:
        #         a, b = env.random_action()
        #         s, _, _ = env.step(np.array([a]), np.zeros(4))
        #         local_steps += 1
        #         state_record.append(s)
        #     while True:
        #         if random.random() < ETA:
        #             action_vehicle = agent_vehicle.select_action(torch.Tensor(state_record[-20:]), ounoise_vehicle,
        #                                                          param_noise_vehicle)[:, -1, :]
        #             # print('rl', action_vehicle.shape)
        #             action_attacker = agent_attacker.select_action(torch.Tensor(state_record[-20:]), ounoise_attacker,
        #                                                            param_noise_attacker)[:, -1, :]
        #         else:
        #             action_vehicle = torch.Tensor(
        #                 [policy_vehicle.predict(state_record[-1].reshape(-1, 4)) / policy_vehicle.predict(
        #                     state_record[-1].reshape(-1, 4)).sum()])[0]
        #             action_attacker = torch.Tensor(
        #                 [policy_attacker.predict(state_record[-1].reshape(-1, 4))])[0]
        #
        #         action_vehicle = action_vehicle.numpy()[0] / action_vehicle.numpy()[0].sum()
        #         action_attacker = action_attacker.numpy()[0]
        #         next_state, reward, done = env.step(action_vehicle, action_attacker)
        #         # distance_file.write('The distance is ' + str(env.d) + '\n')
        #         evaluate_reward += reward
        #
        #         if done:
        #             print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode,
        #                                                                                            total_numsteps,
        #                                                                                            evaluate_reward,
        #                                                                                            np.mean(rewards[-10:])))
        #             # reward_file.write("Episode: {}, total numsteps: {}, reward: {}, average reward: {}\n".format(i_episode,
        #             #                                                                                              total_numsteps,
        #             #                                                                                              evaluate_reward,
        #             #                                                                                              np.mean(rewards[-10:])))
        #             break
        #         # writer.add_scalar('reward/test', episode_reward, i_episode)
        # reward_file.close()
        # attack_file.close()
        # weight_file.close()
        # distance_file.close()
    env.close()
    reward_data.to_csv(suffix+'.csv', index=False)
    res_data.to_csv(suffix+'.csv', index=False)

    # save model
    agent_vehicle.save_model('vehicle_'+suffix)
    agent_attacker.save_model('attacker_'+suffix)

    policy_attacker.save('models/attacker_'+suffix+'.h5')
    policy_vehicle.save('models/vehicle_'+suffix+'.h5')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                        help='discount factor for model (default: 0.001)')
    parser.add_argument('--ou_noise', type=bool, default=True)
    parser.add_argument('--param_noise', type=bool, default=False)
    parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                        help='initial noise scale (default: 0.3)')
    parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                        help='final noise scale (default: 0.3)')
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=4, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size (default: 128)')
    parser.add_argument('--num_steps', type=int, default=100000, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                        help='number of episodes (default: 1000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                        help='number of episodes (default: 128)')
    parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                        help='model updates per simulator step (default: 5)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--NashMode', type=int, default=0, metavar='N',
                        help='Use Nash Mode, 0 means ETA=1, 1 means ETA=0.5, 2 means decayed ETA (default: 0)')
    parser.add_argument('--NonZeroRC', type=bool, default=False, metavar='N',
                        help='RC\'s value is 0 or 100 (default: 0)')
    parser.add_argument('--AttackMode', type=int, default=1, metavar='N',
                        help='Attack mode, 1 means a1, 2 means a3,a4, 4 means fix a1=1, else a1,a2,a3,a4 (default: 1)')
    parser.add_argument('--RewardMode', type=int, default=3, metavar='N',
                        help='Reward mode (default: 3)')
    # python3 dqn_RC_torch.py --Nash True
    from DnsCarFollowENV2 import VehicleFollowingENV

    args = parser.parse_args()
    env = VehicleFollowingENV(args)


    if args.NonZeroRC:
        RC = 100
    else:
        RC = 0
    print("""
Environment Initializing...
The initial head car velocity is {}
The initial safe distance is     {}
The Nash Eq* Factor RC is        {}
The Reward Calculation Mode is   {}
The Attack Mode is               {}
The Nash Mode is                 {}
""".format(env.v_head, env.d0, RC, env.reward_mode, env.attack_mode, args.NashMode))
    # writer = SummaryWriter()
    # logfile = open('error.log', 'a')
    fit_nash()
    # try:
    #     fit_nash()
    # except Exception as e:
    #     logfile.write(str(datetime.datetime.now()) + ' ' + str(e))
    #     logfile.close()
    # visualization()
