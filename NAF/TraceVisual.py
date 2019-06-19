import cubic_spline_planner
import scipy.linalg as la
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import argparse
import numpy as np
import re
from PIL import Image


# parameters
dt = 0.1  # time tick[s]
R = 50.  # visual trace circle R
v_head = 20/3.6
X_pos = np.linspace(-R, R, 500)
Y_pos = np.sqrt(R**2 - X_pos**2)
X_neg = np.linspace(R, -R, 500)
Y_neg = -np.sqrt(R**2 - X_neg**2)
X = np.hstack([X_pos, X_neg])
Y = np.hstack([Y_pos, Y_neg])

plt.plot(X, Y, "-g", linewidth=15)
plt.plot(1.1*X, 1.1*Y, "-k", linewidth=15)
plt.plot(0.9*X, 0.9*Y, "-g", linewidth=15)


class State:

    def __init__(self, x=0.0, y=R):
        assert x**2 + y**2 == R**2, "invalid input position"
        self.x = x
        self.y = y
        self.get_theta()

    def reset(self):
        self.theta += np.random.random()*np.pi/2
        if self.theta > np.pi:
            self.theta -= np.pi
        self.get_xy(self.theta)

    def get_theta(self):
        if self.x == 0 and self.y > 0:
            self.theta = np.pi/2
        elif self.x == 0 and self.y < 0:
            self.theta = -np.pi/2
        elif self.x < 0 and self.y >= 0:
            self.theta = np.arctan(self.y/self.x) + np.pi
        elif self.x < 0 and self.y <= 0:
            self.theta = np.arctan(self.y/self.x) - np.pi
        else:
            self.theat = np.arctan(self.y/self.x)
        return self.theta

    def get_xy(self, theta=None):
        self.theta = theta if theta else self.theta
        if self.theta == np.pi/2:
            self.x = 0
            self.y = R
        elif self.theta == -np.pi/2:
            self.x = 0
            self.y = -R
        else:
            self.x = R*np.cos(self.theta)
            self.y = R*np.sin(self.theta)

def update(state, distance_prev=0., distance_now=0., step_size=1):
    # 输入上一时刻的跟车距离和此时的跟车距离, 计算当前位置, 也适用于前车

    alpha = (5 + v_head*dt*step_size - (distance_now-distance_prev))/R
    theta = state.get_theta() + alpha

    if theta <= -np.pi:
        theta += 2*np.pi
    if theta > np.pi:
        theta -= 2*np.pi

    state.get_xy(theta)

def plot(state1, state2, num, local_step, distance, episode):
    if num < 10:
        num = '000'+str(num)
    elif num < 100:
        num = '00'+str(num)
    elif num < 1000:
        num = '0'+str(num)
    plt.subplots(1, figsize=(8,8))
    plt.title('Car Following Result, Time = ' + str(local_step*5) +
              's '+'Distance = '+str(distance)[:5]+'m\n'+'Episode = ' +
              str(episode))
    plt.plot(X, Y, "-g", linewidth=15)
    plt.plot(X, Y, "-k", linewidth=1)


    plt.plot(state1.x, state1.y, "^b", markersize=12, label="Vehicle_self")
    plt.plot(state2.x, state2.y, "^r", markersize=12, label="Vehicle_head")
    # plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig('Fig/a'+str(num)+'.png')
    plt.close()
    print('Fig/a'+str(num)+'.png saved!!!')
    # plt.show()


def initial_distance(state, state2):
    alpha = 25 / R
    theta = state.get_theta() + alpha
    state2.get_xy(theta)


def get_distance_data(file, bound=np.inf):
    assert 'txt' in file, 'Not supported file type..'
    i = 0
    data = [[]]
    with open(file) as f:
        for line in f.readlines():
            distance = re.findall('\d+\.\d{3}', line)
            if i > bound:
                break
            if distance:
                value = float(distance[0])
                if value < 1 or value > 40:
                    data[-1].append(value)
                    data.append([])
                else:
                    data[-1].append(value)
                i += 1
    return data

def get_distance_from_csv(file, bound=np.inf):
    i = 0
    data = [[]]
    distances = pd.read_csv(file)
    for distance in distances['Eva_distance'].values:
        if i > bound:
            break
        value = float(distance)
        if value < 20 or value > 30:
            data[-1].append(value+(value-25)*4)
            data.append([])
        else:
            data[-1].append(value+(value-25)*4)
        i += 1
    return data

def gen_gif():
    import os
    files = []
    for _,_,file in os.walk('Fig'):
        files.append(file)
    im = Image.open("Fig/a0000.png")
    images = []
    for file in sorted(files[0]):
        file = 'Fig/'+file
        if len(file)==len("Fig/a0000.png") and file[-4:] == '.png':
            # print(file)
            images.append(Image.open(file))
    im.save('Gif/res2.gif', save_all=True, append_images=images, loop=1, duration=0.5)


def main():

    max_episode = 10
    num_episode = 0
    num = 0
    step_size = 10  # 表示每个episode内选取距离的步长

    car_self = State()
    car_head = State()
    try:
        distance_data_list = get_distance_data(file)
    except:
        distance_data_list = get_distance_from_csv(file)

    distance_prev = None


    for distance_data in distance_data_list[::len(distance_data_list)//max_episode]:
        local_step = 0
        print(len(distance_data_list))
        for distance in distance_data[::step_size]:
            if not distance_prev:
                initial_distance(car_self, car_head)
                print('Start Episode {}'.format(len(distance_data_list)//max_episode*(num_episode)))

            else:
                local_step += 1
                update(state=car_self, distance_prev=distance_prev, distance_now=distance, step_size=step_size//10)
                update(car_head, step_size=step_size//10)
                plot(car_self, car_head, num, local_step,
                     distance, len(distance_data_list)//max_episode*(num_episode)*10)
                num += 1

            if num_episode > max_episode:
                break

            distance_prev = distance

        car_self.reset()
        distance_prev = None
        num_episode += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distance Visual Parser')
    parser.add_argument('--file', type=str, default='', metavar='G', help='Input Distance Filename')
    args = parser.parse_args()
    file = args.file
    path = os.getcwd()
    os.system('rm /Users/mac/Desktop/深度强化学习/Hw/4-nfsp/RL_robust_control/NAF/Fig/*')
    main()
    gen_gif()


