"""Compatibility shim: maps flatland v3 RailAgentStatus to flatland v4 TrainState."""
from flatland.envs.agent_utils import TrainState

# In flatland v4, on-map states are split (was a single ACTIVE in v3)
_ON_MAP_STATES = frozenset([TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION])


class _ActiveSentinel:
    """Compares equal to any on-map train state (MOVING, STOPPED, MALFUNCTION)."""
    def __eq__(self, other):
        return other in _ON_MAP_STATES

    def __ne__(self, other):
        return other not in _ON_MAP_STATES

    def __hash__(self):
        return hash('_ACTIVE_SENTINEL')


class RailAgentStatus:
    WAITING = TrainState.WAITING
    READY_TO_DEPART = TrainState.READY_TO_DEPART
    ACTIVE = _ActiveSentinel()
    DONE = TrainState.DONE
    DONE_REMOVED = TrainState.DONE  # merged into DONE in flatland v4
