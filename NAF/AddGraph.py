from matplotlib import pyplot as plt
import re
import numpy as np
import pandas as pd
from DnsCarFollowENV2 import VehicleFollowingENV
import random
import argparse

np.random.seed(1234)
random.seed(1234)

def plotReward(filename, step=10):
    reward = []
    episode = []
    with open(filename) as f:
        for line in f.readlines():
            if step != 10:
                res = re.findall('Episode (.*) ends, instant reward is (.*)', line)
            else:
                res = re.findall('Episode: (.*), total numsteps: .*, reward: .*, average reward: (.*)', line)
            if len(res) != 0:
                if float(res[0][1]) < -200000:
                    reward.append(-200000)

                else:
                    reward.append(float(res[0][1]))

                episode.append(float(res[0][0]))

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    if step != 10:
        plt.title("Vehicle's total reward changes by 1 episode", fontsize=12)
    else:
        plt.title("Vehicle's total reward changes by 10 episodes", fontsize=12)
    plt.plot(episode, reward)
    # plt.ylim(0, 7000)
    plt.show()


def plotAction(filename):
    w1s = []
    w2s = []
    w3s = []
    w4s = []
    step = []
    i = 0

    with open(filename) as f:
        for line in f.readlines():
            res = re.findall('\[\s*(\S*)\s*(\S*)\s*(\S*)\s*(\S*)\s*\]', line)
            if len(res) != 0:
                (w1, w2, w3, w4) = res[0]
                w1s.append(float(w1))
                w2s.append(float(w2))
                w3s.append(float(w3))
                w4s.append(float(w4))
                step.append(i)
                i += 1

    i = 0
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    # k = int(len(step)/50)
    k = 1000
    while i < len(step):
        if "weight" in filename:
            total = 1
            # total = np.average(w1s[i:i+k]) + np.average(w2s[i:i+k]) + np.average(w3s[i:i+k]) + np.average(w4s[i:i+k])
        else:
            total = 1
        y1.append(np.average(w1s[i:i + k]) / total)
        y2.append(np.average(w2s[i:i + k]) / total)
        y3.append(np.average(w3s[i:i + k]) / total)
        y4.append(np.average(w4s[i:i + k]) / total)
        i += k
    plt.xlabel('Total step/' + str(k))
    if "weight" in filename:
        plt.ylim(min(y1 + y2 + y3 + y4) * 0.99, max(y1 + y2 + y3 + y4) * 1.01)
        plt.ylabel('Weight', fontsize=12)
        plt.title('Weight changes with total step', fontsize=12)
        plt.legend(['w1', 'w2', 'w3', 'w4'], fontsize=8)

    else:
        plt.ylabel('Attack', fontsize=12)
        plt.title('Attack changes with total step', fontsize=12)
        plt.legend(['a1', 'a2', 'a3', 'a4'], fontsize=8)

    plt.step(range(len(y1)), y1, c='r')
    plt.step(range(len(y1)), y2, c='k')
    plt.step(range(len(y1)), y3, c='b')
    plt.step(range(len(y1)), y4, c='g')

    # plt.show()


def plotDistance(filename):
    dis = []
    i = 0
    legends = []
    with open(filename) as f:
        for line in f.readlines():
            res = re.findall('The distance is (.*)', line)
            if len(res) == 0:
                i += 1
                dis.append([])
                legends.append('episode' + str(i * 10))
            else:
                dis[-1].append(float(res[0]))
    print(len(dis))
    for dis_data in dis[19::10]:
        plt.step(range(len(dis_data)), dis_data)
    plt.legend(legends[19::10], fontsize=8)
    plt.xlabel('Number step in episode.', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.title('Distance changes within one episode', fontsize=12)
    # plt.show()


def parseActionCSV(filename, action, k=None):
    try:
        AttackMode = int(re.findall('AttackMode_(\d)', filename)[0])
    except:
        AttackMode = 3
    data = pd.read_csv(filename)
    name = action
    if name != 'Eva_distance':
        data['w1'] = data[action].apply(lambda x: float(x.replace('[', '').replace(']', '').split()[0]))
        data['w2'] = data[action].apply(lambda x: float(x.replace('[', '').replace(']', '').split()[1]))
        data['w3'] = data[action].apply(lambda x: float(x.replace('[', '').replace(']', '').split()[2]))
        data['w4'] = data[action].apply(lambda x: float(x.replace('[', '').replace(']', '').split()[3]))
        w1 = data.w1.values
        w2 = data.w2.values
        w3 = data.w3.values
        w4 = data.w4.values
        y1 = []
        y2 = []
        y3 = []
        y4 = []
        i = 0
        total = 1
        if action == 'Attack':
            p = 1
        else:
            p = 1
        # k = int(w1.shape[0] / 50)
        if not k:
            k = w1.shape[0]//50
        else:
            k = k
        print('k=', k)
        while i < w1.shape[0]:
            y1.append(p*np.average(w1[i:i + k]) / total)
            y2.append(p*np.average(w2[i:i + k]) / total)
            y3.append(p*np.average(w3[i:i + k]) / total)
            y4.append(p*np.average(w4[i:i + k]) / total)
            i += k
        # plt.subplots(1, figsize=(10, 8))
        plt.xlabel('Total step/' + str(k), fontsize=12)
        if action == 'Weight':
            plt.ylim(min(y1 + y2 + y3 + y4) * 0.99, max(y1 + y2 + y3 + y4) * 1.01)
        plt.ylabel(action, fontsize=12)
        plt.title(action + ' changes with total step', fontsize=12)
        AttackMode = 2
        if action == 'Attack':
            if AttackMode == 1:
                y2 = y3 = y4 = np.zeros_like(y1)
            elif AttackMode == 2:
                y1 = y2 = np.zeros_like(y3)
            elif AttackMode == 4:
                y1 = np.ones_like(y2)
                y2 = y3 = y4 = np.zeros_like(y1)

        plt.step(range(len(y1)-1), y1[:-1], c='r')
        plt.step(range(len(y1)-1), y2[:-1], c='k')
        plt.step(range(len(y1)-1), y3[:-1], c='b')
        plt.step(range(len(y1)-1), y4[:-1], c='g')
        # plt.plot(range(len(y1) - 1), y1[:-1], c='r')
        # plt.plot(range(len(y1) - 1), y2[:-1], c='k')
        # plt.plot(range(len(y1) - 1), y3[:-1], c='b')
        # plt.plot(range(len(y1) - 1), y4[:-1], c='g')
        if action == 'Weight':
            plt.legend(['w1', 'w2', 'w3', 'w4'])
        else:
            plt.legend(['a1', 'a2', 'a3', 'a4'])

        # plt.show()
    else:
        data[action] = data[action].astype('float32')
        w = [[]]
        legends = []
        i = 0
        for distance in data[action].values:
            if distance <= 20 or distance >= 30:
                w[-1].append(distance)
                w.append([])
                legends.append('episode' + str(i))
                i += 1
            else:
                w[-1].append(distance)
        print(len(w))
        for dis_data in w[::int(len(w)/10)]:
            plt.step(range(len(dis_data)), dis_data)
        plt.legend(legends[::int(len(w)/10)], fontsize=8)
        plt.xlabel('Number step in episode.', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.title('Distance changes within one episode', fontsize=12)
        # plt.show()


def subspace(x):
    x = re.sub('\[\s+', '[', x)
    x = re.sub('\s+\]', ']', x)
    x = re.sub('\s+', ',', x)
    return x


def ResolveDistance(filename, prefix=''):
    env = VehicleFollowingENV()
    env.reset()
    data = pd.read_csv(filename)
    data['Weight_array'] = data['Weight'].apply(lambda x: eval('np.array(' + subspace(x) + ')'))
    data['Attack_array'] = data['Attack'].apply(lambda x: eval('np.array(' + subspace(x) + ')'))
    i = 0
    weights = data['Weight_array'].values
    attacks = data['Attack_array'].values
    distances = [[]]
    distance_data = []
    legends = []
    while i < data['Weight_array'].shape[0]:
        state, _, is_done = env.step(weights[i], attacks[i])
        distance_data.append(env.d)
        if is_done:
            distances[-1].append(env.d)
            distances.append([])
            legends.append('episode' + str(len(distances) - 1))
            env.reset()
        else:
            distances[-1].append(env.d)
        i += 1
    data['Eva_distance'] = pd.Series(distance_data)
    data[['Weight', 'Attack', 'Eva_distance']].to_csv(prefix + filename, index=False)
    for dis_data in distances[0::int(len(distances)/10)*10]:
        plt.step(range(len(dis_data)), dis_data)
    plt.legend(legends[0::int(len(distances)/10)*10], fontsize=8)
    plt.xlabel('Number step in episode.', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.title('Distance changes within one episode', fontsize=12)


def from3File():
    plt.figure(figsize=(12, 6))
    # plt.subplot(1, 3, 1)
    # plotDistance(actionfile)
    plt.subplot(1, 2, 1)
    plotAction(actionfile.replace('Distance', 'vehicle_weight'), 20000)
    plt.subplot(1, 2, 2)
    plotAction(actionfile.replace('Distance', 'attacker_action'), 20000)




def main():
    plt.figure(figsize=(18, 6))
    # plt.subplot(1,3,1)
    # plt.xticks([])
    # plt.yticks([])
    # plt.text(0.05, 0.5,'_'.join(actionfile.replace('bacon', 'beacon').split('_')[4:7])[:-4]+'Episode', fontsize=16)
    #
    # ax = plt.gca()
    # ax.axes.get_yaxis().set_visible(False)
    # ax.axes.get_xaxis().set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    try:
        plt.subplot(1, 3, 1)
        parseActionCSV(actionfile, 'Eva_distance', args.k)
    except Exception as e:
        print(e)
        ResolveDistance(actionfile)

    plt.subplot(1, 3, 2)
    parseActionCSV(actionfile, 'Weight', args.k)
    plt.subplot(1, 3, 3)
    parseActionCSV(actionfile, 'Attack', args.k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run on server signal')
    parser.add_argument('--server', type=bool, default=False, metavar='G',
                        help='if this code runs in server')
    parser.add_argument('--file', type=str, default='', metavar='G',
                        help='if this code runs in server')
    parser.add_argument('--k', type=int, default=None, metavar='G',
                        help='Step size of graph')

    args = parser.parse_args()
    server = args.server
    actionfile = args.file

    # try:
    #     k = int(re.findall('_(\d*000)[_\.]', actionfile)[0])
    # except:
    #     k = 100


    # from3File()
    main()
    if server:
        plt.savefig(actionfile+'.jpg')
    else:
        plt.show()
    # plt.savefig(actionfile.replace('Distance', '')[:-4]+'.jpg')
