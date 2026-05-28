import numpy as np

class HopfieldNetwork:
    def __init__(self, n_neurons=45):
        """
        Initializes a Hopfield Network.
        
        Parameters:
            n_neurons (int): Number of neurons in the network (default is 45 for 9x5 grid).
        """
        self.n_neurons = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))
        
    def train(self, patterns):
        """
        Trains the Hopfield Network using Hebbian learning.
        
        Parameters:
            patterns (list of np.ndarray): List of bipolar pattern vectors to store.
        """
        self.W = np.zeros((self.n_neurons, self.n_neurons))
        for p in patterns:
            # Enforce flat 1D array of correct size
            p_flat = np.array(p).flatten()
            if len(p_flat) != self.n_neurons:
                raise ValueError(f"Pattern size {len(p_flat)} does not match network size {self.n_neurons}")
            self.W += np.outer(p_flat, p_flat)
            
        # Normalize weights by number of neurons and zero-out diagonal
        self.W /= self.n_neurons
        np.fill_diagonal(self.W, 0)
        
    def energy(self, state):
        """
        Calculates the energy of the current network state.
        E = -0.5 * sum_{i != j} W_ij * s_i * s_j
        
        Parameters:
            state (np.ndarray): State vector of bipolar values (-1 or 1).
        
        Returns:
            float: Energy value.
        """
        s = np.array(state).flatten()
        return -0.5 * np.dot(s, np.dot(self.W, s))
        
    def predict_asynchronous(self, initial_state, max_sweeps=100, record_history=False):
        """
        Reconstructs the stored pattern using asynchronous update dynamics.
        In each sweep, all neurons are updated in a random permutation order.
        
        Parameters:
            initial_state (np.ndarray): Corrupted or incomplete input pattern.
            max_sweeps (int): Maximum number of full updates (sweeps) through all neurons.
            record_history (bool): If True, records the state and energy at each neuron update.
            
        Returns:
            recovered_state (np.ndarray): Final stable state.
            converged (bool): True if stable state was reached before limit, False otherwise.
            history (dict): Dictionary with keys 'states' and 'energies' tracking updates.
        """
        state = np.array(initial_state).flatten().copy()
        
        history_states = [state.copy()]
        history_energies = [self.energy(state)]
        
        converged = False
        
        for sweep in range(max_sweeps):
            # Asynchronous update order is a random permutation of neuron indices
            order = np.random.permutation(self.n_neurons)
            changed = False
            
            for neuron_idx in order:
                # Compute activation sum
                h = np.dot(self.W[neuron_idx], state)
                
                # Bipolar threshold activation function:
                # if h > 0 -> +1
                # if h < 0 -> -1
                # if h == 0 -> keep current state
                new_val = 1 if h > 0 else (-1 if h < 0 else state[neuron_idx])
                
                if new_val != state[neuron_idx]:
                    state[neuron_idx] = new_val
                    changed = True
                    
                    if record_history:
                        history_states.append(state.copy())
                        history_energies.append(self.energy(state))
            
            # If no neuron changed in the entire sweep, the network has converged
            if not changed:
                converged = True
                break
                
        # If history was not requested, make sure we have at least final state
        if not record_history:
            history_states.append(state.copy())
            history_energies.append(self.energy(state))
            
        history = {
            'states': history_states,
            'energies': history_energies,
            'sweeps_completed': sweep + 1
        }
        
        return state, converged, history

    def predict_synchronous(self, initial_state, max_iterations=100, record_history=False):
        """
        Reconstructs the stored pattern using synchronous update dynamics.
        All neurons are updated simultaneously.
        
        Parameters:
            initial_state (np.ndarray): Corrupted or incomplete input pattern.
            max_iterations (int): Maximum number of iterations.
            record_history (bool): If True, records state and energy at each iteration.
            
        Returns:
            recovered_state (np.ndarray): Final state.
            converged (bool): True if stable state was reached, False otherwise.
            history (dict): Dictionary tracking the states and energies.
        """
        state = np.array(initial_state).flatten().copy()
        
        history_states = [state.copy()]
        history_energies = [self.energy(state)]
        
        converged = False
        
        for iteration in range(max_iterations):
            h = np.dot(self.W, state)
            
            # Synchronous update
            new_state = np.zeros_like(state)
            for i in range(self.n_neurons):
                new_state[i] = 1 if h[i] > 0 else (-1 if h[i] < 0 else state[i])
                
            if np.array_equal(new_state, state):
                converged = True
                break
                
            state = new_state.copy()
            
            if record_history:
                history_states.append(state.copy())
                history_energies.append(self.energy(state))
                
        if not record_history:
            history_states.append(state.copy())
            history_energies.append(self.energy(state))
            
        history = {
            'states': history_states,
            'energies': history_energies,
            'iterations_completed': iteration + 1
        }
        
        return state, converged, history
