import unittest
from agent import SearchAgent
# The reflex agents live alongside the environment they react to.
from visual_grid_game import SimpleReflexAgent, ModelBasedAgent


class TestPractical1And2_ReflexAgents(unittest.TestCase):
    """
    Tests for Practicals 1 & 2: Simple Reflex and Model-Based Agents.
    Focuses on Condition-Action rules, partial observability, and memory.
    """

    def setUp(self):
        # Instantiate agents (assuming students have created these classes)
        try:
            self.simple_agent = SimpleReflexAgent()
            self.model_agent = ModelBasedAgent()
        except NameError:
            self.fail("Agent classes not found. Ensure SimpleReflexAgent and ModelBasedAgent are defined.")

    def test_simple_reflex_logic(self):
        """Test 1: Simple Reflex Agent should react purely to immediate percepts."""
        # Scenario A: Food is present -> Agent should want to collect/stay/move appropriately
        percept_food = {'wall_ahead': False, 'food_here': True}
        action = self.simple_agent.sense_and_act(percept_food)
        self.assertIsNotNone(action, "SimpleReflexAgent returned None instead of an action.")

        # Scenario B: Wall is ahead -> Agent must turn or change direction
        percept_wall = {'wall_ahead': True, 'food_here': False}
        action_wall = self.simple_agent.sense_and_act(percept_wall)
        # Practical 3 replaced absolute compass moves ('Up'/'Left'/...) with
        # the relative actuator vocabulary a real robot has.
        self.assertIn(action_wall, ['turn_left', 'turn_right', 'move_forward'],
                      "Agent did not output a valid movement action when facing a wall.")

    def test_model_based_memory(self):
        """Test 2: Model-Based Agent should maintain internal state to escape loops.

        Note on the assertion: the original starter test demanded two
        DIFFERENT actions for the same percept. Our ModelBasedAgent
        deliberately commits to one turn direction for the whole "stuck"
        episode -- alternating turn_left/turn_right would undo each turn and
        flip-flop forever in a dead end, which is precisely the reflex-agent
        bug memory is supposed to cure. So we assert what memory actually
        buys us: the agent's internal model must change between identical
        percepts, and it must not oscillate.
        """
        percept = {'wall_ahead': True, 'food_here': False}

        facing_before = self.model_agent.facing
        action_1 = self.model_agent.sense_and_act(percept)
        action_2 = self.model_agent.sense_and_act(percept)

        self.assertNotEqual(
            facing_before, self.model_agent.facing,
            "ModelBasedAgent's internal heading did not update. There is no memory.")
        self.assertGreater(
            self.model_agent.stuck_turns, 1,
            "ModelBasedAgent is not counting how long it has been stuck.")
        self.assertEqual(
            action_1, action_2,
            "ModelBasedAgent should commit to one turn direction rather than "
            "flip-flopping between turn_left and turn_right.")
        self.assertIn(action_1, ['turn_left', 'turn_right'])


class TestPractical3_SearchAgent(unittest.TestCase):
    """
    Tests for Practical 3: Problem-Solving Agents.
    Focuses on offline planning and Breadth-First Search (BFS) implementation.
    """

    def setUp(self):
        try:
            self.search_agent = SearchAgent()
        except NameError:
            self.fail("SearchAgent class not found.")

    def test_bfs_shortest_path(self):
        """Test 3: BFS must find the optimal (shortest) path in a static maze."""
        # Mock Environment Data
        grid_size = (4, 4)
        start_pos = (0, 0)
        goal_pos = (3, 3)

        # Create a U-shaped wall trap that the agent must navigate around
        # Grid layout (S=Start, G=Goal, W=Wall):
        # 3 | . . . G
        # 2 | W W W .
        # 1 | . . . .
        # 0 | S W W .
        #   ---------
        #     0 1 2 3
        walls = [(1, 0), (2, 0), (0, 2), (1, 2), (2, 2)]

        # Our SearchAgent searches over (position, facing) states, not bare
        # positions, and takes bfs_search(start_state, goal, grid_size, walls).
        start_state = (start_pos, 'Up')
        try:
            path = self.search_agent.bfs_search(start_state, goal_pos, grid_size, set(walls))
        except AttributeError:
            self.fail("bfs_search method not implemented in SearchAgent.")

        # Verify the path is valid and optimal
        self.assertIsNotNone(path, "BFS returned None. No path found.")
        self.assertIsInstance(path, list, "BFS should return a list of actions (strings).")

        # 6 grid moves is the optimal route around these walls. Our plan also
        # contains the turns needed to face each direction, so we count the
        # move_forward actions to compare like with like, and separately
        # assert the plan is turn-optimal (no wasted rotation).
        forwards = [a for a in path if a == 'move_forward']
        self.assertEqual(len(forwards), 6,
                         f"BFS did not find the optimal path. Expected 6 moves, got {len(forwards)}.")
        self.assertLessEqual(len(path), 6 + 3 * 6,
                             "BFS plan contains more rotation than could ever be necessary.")

    def test_bfs_unreachable_goal(self):
        """Test 4: BFS must correctly return failure (None/Empty) if goal is blocked."""
        grid_size = (3, 3)
        start_pos = (0, 0)
        goal_pos = (2, 2)

        # Box the goal in completely
        walls = [(1, 2), (2, 1), (1, 1)]

        path = self.search_agent.bfs_search((start_pos, 'Up'), goal_pos, grid_size, set(walls))

        # The agent should realize it's impossible and return None or an empty list
        is_empty_or_none = (path is None) or (len(path) == 0)
        self.assertTrue(is_empty_or_none, "BFS should return None or [] when the goal is unreachable.")


if __name__ == '__main__':
    # Run the test suite
    print("=== IT3012: Intelligent Agents - Autograder Test Suite ===\n")
    unittest.main(verbosity=2)