import numpy as np
from hopfield import HopfieldNetwork
from patterns import PATTERNS

def test_dimensions():
    """Verify that network initialization sets correct neuron and weight dimensions."""
    net = HopfieldNetwork(n_neurons=45)
    assert net.n_neurons == 45
    assert net.W.shape == (45, 45)

def test_hebbian_weights():
    """Verify that training creates symmetric weights with a zero diagonal."""
    net = HopfieldNetwork(n_neurons=45)
    net.train(PATTERNS)
    
    # Check diagonal is zero
    assert np.allclose(np.diag(net.W), 0)
    
    # Check weight symmetry
    assert np.allclose(net.W, net.W.T)
    
    # Check non-trivial weights
    assert np.any(net.W != 0)

def test_noiseless_recovery():
    """Verify that stored patterns are stable states (zero energy change & zero errors)."""
    net = HopfieldNetwork(n_neurons=45)
    net.train(PATTERNS)
    
    for idx, p in enumerate(PATTERNS):
        # Asynchronous reconstruction from noiseless pattern
        recovered_async, converged_async, hist_async = net.predict_asynchronous(p, max_sweeps=10, record_history=True)
        assert np.array_equal(recovered_async, p), f"Pattern {idx+1} (async) was not stable"
        assert converged_async, f"Pattern {idx+1} (async) did not converge"
        assert len(hist_async['states']) == 1, f"Pattern {idx+1} (async) changed state when it should be stable"
        
        # Synchronous reconstruction from noiseless pattern
        recovered_sync, converged_sync, hist_sync = net.predict_synchronous(p, max_iterations=10, record_history=True)
        assert np.array_equal(recovered_sync, p), f"Pattern {idx+1} (sync) was not stable"
        assert converged_sync, f"Pattern {idx+1} (sync) did not converge"
        assert len(hist_sync['states']) == 1, f"Pattern {idx+1} (sync) changed state when it should be stable"

def test_energy_minimization():
    """Verify that energy decreases monotonically during asynchronous reconstruction of noisy patterns."""
    net = HopfieldNetwork(n_neurons=45)
    net.train(PATTERNS)
    
    np.random.seed(42)
    # Take a pattern and add some noise
    p = PATTERNS[0]
    noisy = p.copy()
    flip_indices = np.random.choice(45, 9, replace=False) # 20% noise
    noisy[flip_indices] *= -1
    
    recovered, converged, history = net.predict_asynchronous(noisy, max_sweeps=50, record_history=True)
    
    energies = history['energies']
    
    # Verify that energy decreases or remains constant on each micro-step (update of a single neuron)
    for i in range(1, len(energies)):
        assert energies[i] <= energies[i-1], f"Energy increased at step {i}: {energies[i-1]} -> {energies[i]}"
