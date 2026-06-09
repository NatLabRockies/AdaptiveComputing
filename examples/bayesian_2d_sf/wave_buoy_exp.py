import numpy as np
import matplotlib.pyplot as plt
from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriver

if __name__ == "__main__":
    
    params = [ContinuousVariable(min=0, max=0.8),
              ContinuousVariable(min=5, max=30)]

    ac_driver = ActiveLoopDriver(simulations=[None],
                                   params=params,
                                   surrogate='SMT_GP',
                                   acq_func='maximum_variance')

    # Create meshgrid
    x1 = np.linspace(0, 0.8, 400)
    x2 = np.linspace(5, 30, 400)
    X1, X2 = np.meshgrid(x1, x2)

    x_samples = np.array([[0.125000000000000, 5],
        [0.125000000000000, 10],
        [0.125000000000000, 15],
        [0.125000000000000, 20],
        [0.125000000000000, 25],
        [0.125000000000000, 30],
        [0.250000000000000, 5],
        [0.250000000000000, 10],
        [0.250000000000000, 15],
        [0.250000000000000, 20],
        [0.250000000000000, 25],
        [0.250000000000000, 30],
        [0.500000000000000, 5],
        [0.500000000000000, 10],
        [0.500000000000000, 15],
        [0.500000000000000, 20],
        [0.750000000000000, 5],
        [0.750000000000000, 10],
        [0.750000000000000, 15],
        [0.750000000000000, 20],
    ])
    y_samples = [ 1.49924597815690,
        0.0297820930874535,
        0.523090302029244,
        0.461067561908239,
        3.39154313402351,
        65.0237974630356,
        0.109265420656735,
        0.220662647590603,
        0.381065901924725,
        0.0418166365514344,
        4.21520941813546,
        64.9849602071857,
        0.126730876393988,
        0.00374813300646995,
        0.230723711230097,
        0.272298633495670,
        0.0170868670147425,
        0.0397516869696306,
        0.121758897493229,
        36.1654246976381,
    ]
    y_samples = np.atleast_2d(y_samples).T

    ac_driver.dataset.add_samples(x_samples,y_samples,i_fidelity=0)
    ac_driver.surrogate.train(ac_driver.dataset)

    # Stack into a (N, 2) array of points
    points = np.column_stack([X1.ravel(), X2.ravel()])  # shape (160000, 2)

    # Evaluate surrogate predictions
    Z_val = ac_driver.surrogate.predict_values(points).reshape(X1.shape)
    Z_var = ac_driver.surrogate.predict_variances(points).reshape(X1.shape)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # First panel: original data
    axes[0].scatter(x_samples[:, 0], x_samples[:, 1], c=y_samples.flatten(), cmap='viridis', s=50)
    axes[0].set_title("Original Data")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")

    # Second panel: predicted values
    c1 = axes[1].imshow(Z_val, extent=(x1.min(), x1.max(), x2.min(), x2.max()),
                        origin='lower', aspect='auto')
    axes[1].set_title("Surrogate Predicted Values")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    fig.colorbar(c1, ax=axes[1])

    # Third panel: predicted variances
    c2 = axes[2].imshow(Z_var, extent=(x1.min(), x1.max(), x2.min(), x2.max()),
                        origin='lower', aspect='auto')
    axes[2].set_title("Surrogate Predicted Variances")
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")
    fig.colorbar(c2, ax=axes[2])

    # suggest next point
    x_next = ac_driver.get_next_sample()
    x_next = x_next[0][0] # assume fidelity level is zero
    axes[2].scatter(x_next[0],x_next[1],s=50,color='red')


    plt.tight_layout()
    plt.show()
