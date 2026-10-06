def is_recurrent(model):
    """RecurrentPPO's policy has an LSTM; its predict() takes and returns the state.

    Here, not in learning.policy, so the robot can ask without importing PyTorch.
    """
    return hasattr(getattr(model, 'policy', None), 'lstm_actor')
