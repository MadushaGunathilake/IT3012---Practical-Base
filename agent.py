# agent.py
import random
import math
from collections import deque
import heapq

from knowledge_base import SafetyKB, cell, atom_str


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SearchAgent:
    """Goal-based/planning agent. Instead of reacting cell-by-cell like the
    reflex agents in visual_grid_game.py, it builds an internal (position,
    facing) model of itself -- exactly like ModelBasedAgent does -- and uses
    that model plus the newly-exposed 'grid_size' / 'walls' / 'all_food'
    percept keys to simulate the whole map offline. It then runs BFS, DFS,
    UCS, or A* (informed search, using a heuristic) to compute a complete
    action plan to the nearest food pellet before
    it ever moves, and simply replays that plan one action at a time."""

    FACING_OFFSETS = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}
    TURN_LEFT = {'Up': 'Left', 'Left': 'Down', 'Down': 'Right', 'Right': 'Up'}
    TURN_RIGHT = {'Up': 'Right', 'Right': 'Down', 'Down': 'Left', 'Left': 'Up'}

    def __init__(self, active_algo='BFS', heuristic_type='manhattan', verbose=True):
        self.plan = []
        self.verbose = verbose
        self.active_algo = active_algo  # 'BFS', 'DFS', 'UCS', or 'AStar'
        self.heuristic_type = heuristic_type  # 'manhattan' or 'euclidean', only used by AStar

        # Self-tracked model of the world, built the same way ModelBasedAgent
        # does it: the environment never hands out agent_pos directly.
        self.pos = (0, 0)
        self.facing = 'Up'
        self.last_action = None
        self.target = None  # currently chosen food goal, exposed for the GUI to highlight

        # Cells that search must treat as impassable even though they are
        # physically empty. Plain SearchAgent never fills this in; LogicAgent
        # fills it from its Knowledge Base (Tutorial 05, reachable != feasible).
        self.blocked = set()

    def _update_state(self):
        """Transition model: predict how our own (pos, facing) changed as a
        result of the last action we chose. We only ever plan move_forward
        through cells we already proved are in-bounds and wall-free, so it
        is safe to assume the move always succeeds."""
        if self.last_action == 'move_forward':
            dx, dy = self.FACING_OFFSETS[self.facing]
            self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        elif self.last_action == 'turn_left':
            self.facing = self.TURN_LEFT[self.facing]
        elif self.last_action == 'turn_right':
            self.facing = self.TURN_RIGHT[self.facing]

    def _successors(self, state, grid_size, walls):
        """A state is (pos, facing). Yields (action, next_state, step_cost)."""
        pos, facing = state
        width, height = grid_size

        yield 'turn_left', (pos, self.TURN_LEFT[facing]), 1
        yield 'turn_right', (pos, self.TURN_RIGHT[facing]), 1

        dx, dy = self.FACING_OFFSETS[facing]
        ahead = (pos[0] + dx, pos[1] + dy)
        in_bounds = 0 <= ahead[0] < width and 0 <= ahead[1] < height
        if in_bounds and ahead not in walls and ahead not in self.blocked:
            yield 'move_forward', (ahead, facing), 1

    @staticmethod
    def _reconstruct_path(came_from, goal_state):
        actions = []
        state = goal_state
        while came_from[state] is not None:
            parent_state, action = came_from[state]
            actions.append(action)
            state = parent_state
        actions.reverse()
        return actions

    def bfs_search(self, start_state, goal_pos, grid_size, walls):
        """FIFO frontier -> explores shallowest (fewest-action) nodes first."""
        frontier = deque([start_state])
        reached = {start_state}
        came_from = {start_state: None}

        if start_state[0] == goal_pos:
            return []

        while frontier:
            state = frontier.popleft()
            for action, next_state, _cost in self._successors(state, grid_size, walls):
                if next_state in reached:
                    continue
                reached.add(next_state)
                came_from[next_state] = (state, action)
                if next_state[0] == goal_pos:
                    return self._reconstruct_path(came_from, next_state)
                frontier.append(next_state)
        return []

    def dfs_search(self, start_state, goal_pos, grid_size, walls):
        """LIFO frontier -> explores deepest node first, so it can wander
        down a long branch before backtracking (winding, non-optimal paths)."""
        frontier = [start_state]
        reached = {start_state}
        came_from = {start_state: None}

        if start_state[0] == goal_pos:
            return []

        while frontier:
            state = frontier.pop()
            if state[0] == goal_pos:
                return self._reconstruct_path(came_from, state)
            for action, next_state, _cost in self._successors(state, grid_size, walls):
                if next_state in reached:
                    continue
                reached.add(next_state)
                came_from[next_state] = (state, action)
                frontier.append(next_state)
        return []

    def ucs_search(self, start_state, goal_pos, grid_size, walls):
        """Priority queue ordered by path cost g(n). All step costs here are
        1, so UCS degenerates to BFS's behaviour, but it demonstrates the
        general cost-driven frontier used for weighted graphs."""
        counter = 0  # tie-breaker so heapq never has to compare states directly
        frontier = [(0, counter, start_state)]
        best_cost = {start_state: 0}
        came_from = {start_state: None}
        reached = set()

        while frontier:
            cost, _, state = heapq.heappop(frontier)
            if state in reached:
                continue
            reached.add(state)

            if state[0] == goal_pos:
                return self._reconstruct_path(came_from, state)

            for action, next_state, step_cost in self._successors(state, grid_size, walls):
                new_cost = cost + step_cost
                if next_state not in best_cost or new_cost < best_cost[next_state]:
                    best_cost[next_state] = new_cost
                    came_from[next_state] = (state, action)
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, next_state))
        return []

    def manhattan_distance(self, pos, goal):
        """h(n) = |x1 - x2| + |y1 - y2|. Admissible on a 4-way grid because
        it's exactly the number of moves needed if no walls existed -- it
        never overestimates the true remaining cost."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        """h(n) = sqrt((x1-x2)^2 + (y1-y2)^2), the straight-line distance."""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        """A* over absolute grid positions (Up/Down/Left/Right neighbours),
        independent of facing -- this is the pathfinding search itself, not
        the turn/move_forward actuator plan. f(n) = g(n) + h(n): g(n) is the
        actual steps taken so far, h(n) is the heuristic estimate to the goal."""
        heuristic = self.manhattan_distance if heuristic_type == 'manhattan' else self.euclidean_distance
        width, height = grid_size
        walls = set(walls)

        reached_states = set()
        frontier = [(heuristic(start_pos, goal_pos), 0, start_pos, [])]

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal_pos:
                return path_taken

            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            for direction, (dx, dy) in self.FACING_OFFSETS.items():
                neighbor = (current_pos[0] + dx, current_pos[1] + dy)
                in_bounds = 0 <= neighbor[0] < width and 0 <= neighbor[1] < height
                if not in_bounds or neighbor in walls or neighbor in reached_states:
                    continue
                if neighbor in self.blocked:
                    continue

                g_new = g_cost + 1
                h_new = heuristic(neighbor, goal_pos)
                heapq.heappush(frontier, (g_new + h_new, g_new, neighbor, path_taken + [direction]))

        return []

    def _directions_to_plan(self, directions):
        """astar_search plans in absolute Up/Down/Left/Right steps, but the
        agent can only turn_left/turn_right/move_forward relative to its own
        facing (same actuator vocabulary BFS/DFS/UCS already replay). This
        rotates towards each direction (at most 3 turn_lefts) before moving
        into it, bridging the search result to something sense_and_act can execute."""
        plan = []
        facing = self.facing
        for direction in directions:
            while facing != direction:
                plan.append('turn_left')
                facing = self.TURN_LEFT[facing]
            plan.append('move_forward')
        return plan

    def _closest_food(self, all_food):
        # Never pick a goal cell that has been ruled infeasible -- an
        # unreachable-by-construction goal would make search return an empty
        # plan and the agent would spin in place forever.
        candidates = [food for food in all_food if food not in self.blocked] or list(all_food)
        return min(
            candidates,
            key=lambda food: abs(food[0] - self.pos[0]) + abs(food[1] - self.pos[1]),
        )

    def sense_and_act(self, percept: dict) -> str:
        self._update_state()

        if percept['food_here']:
            self.last_action = 'suck'
            return 'suck'

        if not self.plan:
            all_food = percept['all_food']
            if not all_food:
                self.last_action = 'turn_left'
                return 'turn_left'

            grid_size = percept['grid_size']
            walls = set(percept['walls'])
            goal_pos = self._closest_food(all_food)
            self.target = goal_pos
            start_state = (self.pos, self.facing)

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(start_state, goal_pos, grid_size, walls)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start_state, goal_pos, grid_size, walls)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start_state, goal_pos, grid_size, walls)
            elif self.active_algo == 'AStar':
                directions = self.astar_search(
                    self.pos, goal_pos, walls, grid_size, heuristic_type=self.heuristic_type
                )
                self.plan = self._directions_to_plan(directions)

            if self.verbose:
                manhattan = abs(goal_pos[0] - self.pos[0]) + abs(goal_pos[1] - self.pos[1])
                print(
                    f"[{self.active_algo}] agent at {self.pos} -> target food {goal_pos} "
                    f"| straight-line dist={manhattan} | plan length={len(self.plan)} moves"
                )

            self.plan.append('suck')

        action = self.plan.pop(0)
        self.last_action = action
        return action



class LogicAgent(SearchAgent):
    """Tutorial 05 -- a knowledge-based agent: search + logical validation.

    SearchAgent is purely an *explorer*. It asks "is there a path?" and if A*
    says yes it walks that path, straight over toxic traps, losing 15 points a
    time. It has no way to express "that cell is legal to enter", only "that
    cell is connected to me".

    LogicAgent keeps SearchAgent's whole planning stack and bolts a Knowledge
    Base on top, giving it the Sense -> TELL -> ASK -> Plan -> Act cycle:

      SENSE  read percept['smells_toxin'] at the current cell
      TELL   assert Toxic(Cx_y) into the KB when that sensor fires
      ASK    backward-chain  forall x Toxic(x) ^ NoAntidote(Agent) => Unsafe(x)
             over every cell the planner might want to use
      PLAN   feed the entailed Unsafe cells to search as self.blocked, so the
             planner can no longer even *propose* an infeasible route
      ACT    replay the validated plan one action at a time

    This is Part 1 of the tutorial in code. A trap cell stays Reachable for
    the whole run -- A* would happily route through it -- but once the KB
    entails Unsafe(cell) it stops being Feasible, the current plan is vetoed,
    and search is re-run around it.
    """

    def __init__(self, active_algo='AStar', heuristic_type='manhattan', verbose=True):
        super().__init__(active_algo=active_algo, heuristic_type=heuristic_type,
                         verbose=verbose)
        self.kb = SafetyKB()
        self.vetoed_plans = 0   # how many times logic overruled search
        self.trap_hits = 0      # how many trap entries we paid for

    # -- SENSE / TELL -------------------------------------------------------

    def _tell_percept(self, percept: dict) -> bool:
        """Push sensor readings into the Knowledge Base. Returns True if the
        KB learned something new, which means any existing plan must be
        re-validated against the enlarged KB."""
        if not percept.get('smells_toxin'):
            return False

        self.trap_hits += 1
        if not self.kb.learn_toxic(self.pos):
            return False

        if self.verbose:
            print('  [KB TELL] toxin sensor fired at {} -> assert {}'.format(
                self.pos, atom_str(('Toxic', cell(self.pos)))))
        return True

    # -- ASK ----------------------------------------------------------------

    def _refresh_blocked(self):
        """ASK the KB which cells it now entails Unsafe and hand that set to
        the planner. Only learned toxic cells can possibly be unsafe, so that
        is the candidate set the inference runs over."""
        self.blocked = self.kb.unsafe_cells(self.kb.known_toxic_positions())

    def _plan_is_feasible(self) -> bool:
        """Validate the *remaining* plan against the KB by simulating it
        forward from the current (pos, facing). Search built this plan under
        older knowledge; if it now walks through an Unsafe cell it must be
        discarded. Search proposes, logic disposes."""
        pos, facing = self.pos, self.facing
        for action in self.plan:
            if action == 'turn_left':
                facing = self.TURN_LEFT[facing]
            elif action == 'turn_right':
                facing = self.TURN_RIGHT[facing]
            elif action == 'move_forward':
                dx, dy = self.FACING_OFFSETS[facing]
                pos = (pos[0] + dx, pos[1] + dy)
                if not self.kb.is_feasible(pos):
                    return False
        return True

    def _plan_to(self, goal_pos, grid_size, walls):
        """Run whichever search algorithm is active. self.blocked is already
        consulted inside _successors() and astar_search()."""
        start_state = (self.pos, self.facing)
        if self.active_algo == 'BFS':
            return self.bfs_search(start_state, goal_pos, grid_size, walls)
        if self.active_algo == 'DFS':
            return self.dfs_search(start_state, goal_pos, grid_size, walls)
        if self.active_algo == 'UCS':
            return self.ucs_search(start_state, goal_pos, grid_size, walls)
        directions = self.astar_search(
            self.pos, goal_pos, walls, grid_size, heuristic_type=self.heuristic_type
        )
        return self._directions_to_plan(directions)

    # -- the full cycle -----------------------------------------------------

    def sense_and_act(self, percept: dict) -> str:
        self._update_state()

        if self._tell_percept(percept):
            self._refresh_blocked()
            if self.plan and not self._plan_is_feasible():
                self.vetoed_plans += 1
                if self.verbose:
                    print('  [KB ASK ] remaining plan crosses an Unsafe cell -> '
                          'VETOED, replanning around {}'.format(sorted(self.blocked)))
                self.plan = []

        if percept['food_here']:
            self.last_action = 'suck'
            return 'suck'

        if not self.plan:
            all_food = percept['all_food']
            if not all_food:
                self.last_action = 'turn_left'
                return 'turn_left'

            grid_size = percept['grid_size']
            walls = set(percept['walls'])
            goal_pos = self._closest_food(all_food)
            self.target = goal_pos

            plan = self._plan_to(goal_pos, grid_size, walls)

            if not plan and self.pos != goal_pos:
                # Reachable-but-infeasible deadlock: the KB has walled off
                # every route to this food. Rather than freeze, drop the
                # constraint for this one plan and say so out loud -- an
                # honest "I had to break a rule" beats a silent stall.
                saved, self.blocked = self.blocked, set()
                plan = self._plan_to(goal_pos, grid_size, walls)
                self.blocked = saved
                if plan and self.verbose:
                    print('  [KB WARN] no feasible route to {}; accepting an '
                          'unsafe path under protest'.format(goal_pos))

            self.plan = plan
            if self.verbose:
                print('[{}+KB] agent at {} -> target {} | plan length={} | '
                      'KB knows {} toxic cell(s)'.format(
                          self.active_algo, self.pos, goal_pos, len(self.plan),
                          len(self.kb.known_toxic_positions())))
            self.plan.append('suck')

        action = self.plan.pop(0)
        self.last_action = action
        return action


if __name__ == "__main__":
    # Practical 4, Step 1.1.5 -- Testing Checkpoint: verify the heuristics
    # against a mock start (0, 0) and goal (3, 4) before wiring up A*.
    agent = SearchAgent()
    start, goal = (0, 0), (3, 4)
    print(f"manhattan_distance{start, goal} = {agent.manhattan_distance(start, goal)}  (expected 7)")
    print(f"euclidean_distance{start, goal} = {agent.euclidean_distance(start, goal)}  (expected 5.0)")

    # Tutorial 05 -- Testing Checkpoint: reachable is not the same as feasible.
    # A 3x1 corridor, agent at (0,0) facing Right, food at (2,0), trap at (1,0).
    print("\nTutorial 05 checkpoint -- reachability vs feasibility:")
    logic = LogicAgent(active_algo='BFS', verbose=False)
    logic.pos, logic.facing = (0, 0), 'Right'
    grid, walls = (3, 1), set()

    print("  before TELL:    plan to (2,0) = {}".format(logic._plan_to((2, 0), grid, walls)))

    logic.kb.learn_toxic((1, 0))          # toxin sensor fires on the middle cell
    logic._refresh_blocked()
    print("  KB entails Unsafe(C1_0) = {}".format(not logic.kb.is_feasible((1, 0))))
    print("  after TELL:     plan to (2,0) = {}  <- still reachable, NOT feasible"
          .format(logic._plan_to((2, 0), grid, walls)))

    logic.kb.take_antidote()              # falsify the NoAntidote(Agent) premise
    logic._refresh_blocked()
    print("  after antidote: plan to (2,0) = {}  <- premise broken, feasible again"
          .format(logic._plan_to((2, 0), grid, walls)))
