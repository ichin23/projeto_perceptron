import numpy as np
from perceptron import Perceptron

def test_initialization():
    print("Testing initialization...")
    p = Perceptron(learning_rate=0.01, n_features=3)
    assert len(p.weights) == 4
    assert np.all(p.weights >= 0) and np.all(p.weights <= 1)
    print("✓ Initialization OK")

def test_prediction():
    print("Testing prediction logic...")
    # Manual weights: [w0, w1, w2, w3]
    # x0 is always -1.
    # u = w0*(-1) + w1*x1 + w2*x2 + w3*x3
    initial_weights = [0.5, 0.2, 0.3, 0.1]
    p = Perceptron(initial_weights=initial_weights)
    
    # x = [1, 1, 1]
    # u = 0.5*(-1) + 0.2*(1) + 0.3*(1) + 0.1*(1) = -0.5 + 0.6 = 0.1
    # y = 1 (since u >= 0)
    assert p.predict([1, 1, 1]) == 1
    
    # x = [0, 0, 0]
    # u = 0.5*(-1) + 0 = -0.5
    # y = -1
    assert p.predict([0, 0, 0]) == -1
    print("✓ Prediction OK")

def test_hebbian_update():
    print("Testing Supervised Hebbian Rule update...")
    initial_weights = [0.5, 0.1, 0.1, 0.1]
    eta = 0.1
    p = Perceptron(learning_rate=eta, initial_weights=initial_weights)
    
    x = [1, 1, 1]
    d = -1
    # u = 0.5*(-1) + 0.1 + 0.1 + 0.1 = -0.5 + 0.3 = -0.2
    # y = -1
    # d - y = -1 - (-1) = 0 -> No update
    assert p.train_step(x, d) == 0
    np.testing.assert_array_almost_equal(p.weights, initial_weights)
    
    # Case with error
    d = 1
    # y was -1. d - y = 1 - (-1) = 2
    # w_new = w_old + eta * (2) * x_with_bias
    # x_with_bias = [-1, 1, 1, 1]
    # w0_new = 0.5 + 0.1 * 2 * (-1) = 0.5 - 0.2 = 0.3
    # w1_new = 0.1 + 0.1 * 2 * (1) = 0.1 + 0.2 = 0.3
    assert p.train_step(x, d) == 1
    expected_weights = [0.3, 0.3, 0.3, 0.3]
    np.testing.assert_array_almost_equal(p.weights, expected_weights)
    print("✓ Hebbian Update OK")

if __name__ == "__main__":
    try:
        test_initialization()
        test_prediction()
        test_hebbian_update()
        print("\nALL ALGORITHM TESTS PASSED ✓")
    except AssertionError as e:
        print(f"\nTEST FAILED ✗")
        raise e
