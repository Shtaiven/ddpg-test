#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def plot_results(load_results=''):
    results = np.load(load_results)
    print(f'N episodes: {len(results)}')
    print(f'Final average reward: {results[-1]}')
    plt.title('TD3 ReacherBulletEnv-v0 Training Results')
    plt.xlabel('episode')
    plt.ylabel('average reward')
    plt.plot(results)
    plt.show()

if __name__ == "__main__":
    results_name = "sfujimoto_test/results/TD3_ReacherBulletEnv-v0_0.npy"
    plot_results(results_name)
