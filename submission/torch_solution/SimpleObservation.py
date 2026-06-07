from collections import defaultdict, deque
import numpy as np

from .env.observations.TreeObsForRailEnv import TreeObsForRailEnv
from .env.compat import RailAgentStatus

# TreeObservation to vector of floats
class SimpleObservation(TreeObsForRailEnv):
    def __init__(self, max_depth, neighbours_depth, timetable, deadlock_checker,
            greedy_checker, parallel=False, malfunction_obs=True, eval=False):
        super(SimpleObservation, self).__init__(max_depth=max_depth)
        self.max_depth = max_depth
        self.neighbours_depth = neighbours_depth
        self.n_neighbours = (2 ** (self.neighbours_depth + 1) - 2)
        self.state_sz = (2 ** (max_depth + 1) - 2) * 14 + 7
        self.parallel = parallel
        self.malfunction_obs = malfunction_obs
        self.eval = eval

        self.timetable = timetable
        self.deadlock_checker = deadlock_checker
        self.greedy_checker = greedy_checker

    def reset(self):
        super().reset()
        self.encountered = defaultdict(list)

        self.random_handles = np.empty(0, dtype=np.float32)
        self.last_action = np.empty(0, dtype=np.int64)
        self._deferred_ready = False
        self._ensure_runtime_ready()

    def get_many(self, handles=None, ignore_parallel=False):
        if self.parallel and not ignore_parallel:
            return []
        self._ensure_runtime_ready()
        self.timetable.update()
        self.deadlock_checker.update_deadlocks()
        observations = super().get_many(handles)
        return observations

    def _ensure_runtime_ready(self):
        if self._deferred_ready or self.env.rail is None:
            return
        self.random_handles = np.random.uniform(0, 1, len(self.env.agents))
        self.last_action = -np.ones(len(self.env.agents), dtype=np.int64)
        self.env.distance_map.reset(self.env.agents, self.env.rail)
        self.deadlock_checker.reset(self.env)
        self.greedy_checker.reset(self.env, self.deadlock_checker)
        self.timetable.reset(self.env)
        self._deferred_ready = True

    def get(self, handle):
        if self._get_checks(handle):
            return self._get_internal(handle)
        else:
            return None

    def _get_checks(self, handle):
        if not self.timetable.is_ready(handle):
            return False
        if self.greedy_checker.greedy_position(handle):
            return False
        if self.deadlock_checker.old_deadlock(handle):
            return False
        if not self.malfunction_obs and self.env.agents[handle].malfunction_handler.malfunction_down_counter != 0:
            return False

        return True

    def _get_node(self, handle):
        return super().get(handle)

    def _get_internal(self, handle, node=None):
        if node is None:
            node = super().get(handle)
        observation = node

        self.cur_handle = handle
        self.encountered[handle].clear()
        if observation is None:
            self.cur_dist = float('inf')
        else:
            self.cur_dist = observation.dist_min_to_target

        observation_vector = list()
        # root node is special
        # # add root features here
        observation_vector.extend([
            #  position[0] / 100.0,
            #  position[1] / 100.0,
            self.random_handles[handle],
            self.norm_bool(self.deadlock_checker.is_deadlocked(handle)),
            self.norm_bool(self.env.agents[handle].malfunction_handler.malfunction_down_counter == 0),
            self.norm_dist(self.cur_dist),
            self.norm_bool(self.greedy_checker.on_switch(handle)),
            self.norm_bool(_is_near_next_decision(observation)),
            self.norm_bool(self.env.agents[handle].state == RailAgentStatus.READY_TO_DEPART)
        ])

        self.traverse(observation, self.max_depth, observation_vector)
        return observation_vector

    def _get_agent_position(self, handle):
        agent = self.env.agents[handle]
        if agent.state in (RailAgentStatus.READY_TO_DEPART, RailAgentStatus.WAITING):
            agent_virtual_position = agent.initial_position
        elif agent.state == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.state in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED):
            agent_virtual_position = agent.target
        else:
            agent_virtual_position = agent.initial_position

        return agent_virtual_position

    def norm_bool(self, val):
        return 2 * int(val) - 1

    def norm_dist(self, dist):
        if dist == float('inf'):
            return 0.
        return dist / 100. - 10 # less then zero (for inf)

    def get_features(self, node, parent, lvl):
        if self.max_depth - lvl < self.neighbours_depth:
            self.encountered[self.cur_handle].append(node.first_agent_handle)
        return [
               1 if node.dist_min_to_target >= 0 else -1, # real node  (not after target or after target) # 0
               self.norm_dist(node.dist_other_agent_encountered), # 1
               self.norm_dist(node.dist_to_next_branch), # 2
               self.norm_dist(node.dist_to_unusable_switch), # 3
               self.norm_dist(node.dist_min_to_target) - self.norm_dist(parent.dist_min_to_target) \
                       if node.dist_min_to_target >= 0 else 0, # delta is better isn't it? # 4
               self.norm_dist(node.dist_min_to_target + node.dist_to_next_branch) - self.norm_dist(parent.dist_min_to_target) \
                       if node.dist_min_to_target >= 0 else 0, # 5
               self.norm_bool(node.num_agents_same_direction >= 1), # 6
               self.norm_bool(node.num_agents_same_direction >= 2), # 7
               self.norm_bool(node.num_agents_opposite_direction >= 1), # 8
               self.norm_bool(node.num_agents_opposite_direction >= 2), # 9
               self.norm_bool(node.max_index_oppposite_direction > self.cur_handle), # are we the best? # 10
               self.norm_bool(node.first_agent_not_opposite), # 11
               self.norm_bool(node.max_handle_agent_not_opposite), # 12
               self.norm_bool(node.has_deadlocked_agent), # 13
        ]


    def get_padding_features(self, lvl):
        if self.max_depth - lvl < self.neighbours_depth:
            self.encountered[self.cur_handle].append(-1)
        return [0] * 14

    # in two directions with a lot of trash?
    def traverse(self, node, lvl, observation):

        q = deque()
        q.append((node, lvl))

        while q:
            node, lvl = q.popleft()
            assert lvl > 0
                
            cnt = 0
            if node is not None and node.childs:
                for value in node.childs:
                    if value:
                        cnt += 1
                        observation.extend(self.get_features(value, node, lvl))
                        if lvl - 1 != 0:
                            q.append((value, lvl - 1))

            assert(cnt <= 2)
            # cnt != 2 in case of target nearby?
            # TODO probably rewriting TreeObs can help
            for i in range(cnt, 2):
                observation.extend(self.get_padding_features(lvl))
                if lvl - 1 != 0:
                    q.append((None, lvl - 1))


def _is_near_next_decision(node):
    next_node = None
    if node is None or not node.childs:
        return False

    for value in node.childs:
        if value:
            if next_node is not None:
                return False # on switch
            next_node = value

    return next_node.dist_to_next_branch == 1 or next_node.dist_to_unusable_switch == 1


