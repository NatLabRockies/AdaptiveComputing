# This script demonstrates basic HeroDataset functionality for task queue management
# Simple introduction to Hero concepts using local worker processing
import numpy as np

import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.datasets import HeroDataset

def func_1d(x):
    return (x-3)**2

if __name__ == '__main__':
    
    params = [ContinuousVariable(min=0.7, max=2.0)]
    # For simple local processing, we use a single local "machine" name
    machine_names = ['local']
    dataset = HeroDataset(params, machine_names, n_fidelity=1, blocking=False)
    # queue hero samples at the given x_data values. No initial guess provided.
    dataset.add_samples(np.array([[1.1]]),None,0)
    dataset.add_samples(np.array([[1.5],[1.8]]),None,0)
    # add samples with specified x_data and y_data. No hero queueing used.
    dataset.add_samples_nohero(np.array([[1.2],[1.3],[1.4]]),np.array([[6.2],[7.2],[8.2]]),0)
    # queue hero samples at the given x_data values. Initial guesses for y_data provided.
    dataset.add_samples(np.array([[1.6],[1.9]]),np.array([[4.2],[5.2]]),0)
    dataset.add_samples(np.array([[1.1]]),np.array([[1.2]]),0)
    print(f'_x_data = {dataset._x_data}')
    print(f'_y_data = {dataset._y_data}')
    print(f'_unmasked_data = {dataset._unmasked_data}')
    print(f'_hero_todo = {dataset._hero_todo}')
    dataset.hero_wait_for_data()
    print(f'_x_data = {dataset._x_data}')
    print(f'_y_data = {dataset._y_data}')
    print(f'_unmasked_data = {dataset._unmasked_data}')
    print(f'_hero_todo = {dataset._hero_todo}')
