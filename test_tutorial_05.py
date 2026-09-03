# test_tutorial_05.py
"""Tutorial 05 test suite -- every part of the tutorial, asserted.

Run:  python -m unittest -v test_tutorial_05.py
"""

import unittest

from agent import LogicAgent, SearchAgent
from knowledge_base import (
    KnowledgeBase,
    SafetyKB,
    cell,
    clause_str,
    negate,
    resolution_prove,
    resolve,
    to_cnf,
    unify,
)


class TestPart1ReachabilityVsFeasibility(unittest.TestCase):
    """Part 1: a state can be reachable and still infeasible."""

    def setUp(self):
        self.kb = KnowledgeBase()
        self.kb.tell_rule([('DarkMap',), ('BloodseekerMissing',)],
                          ('Unsafe', 'River_Rune'))

    def test_both_premises_true_entails_unsafe(self):
        self.kb.tell_fact(('DarkMap',))
        self.kb.tell_fact(('BloodseekerMissing',))
        self.assertTrue(self.kb.ask(('Unsafe', 'River_Rune')))

    def test_one_premise_false_does_not_entail_unsafe(self):
        # A Ward is placed, so DarkMap is never asserted. The rule cannot fire.
        self.kb.tell_fact(('BloodseekerMissing',))
        self.assertFalse(self.kb.ask(('Unsafe', 'River_Rune')))

    def test_retracting_a_premise_restores_feasibility(self):
        self.kb.tell_fact(('DarkMap',))
        self.kb.tell_fact(('BloodseekerMissing',))
        self.assertTrue(self.kb.ask(('Unsafe', 'River_Rune')))
        self.kb.retract(('DarkMap',))
        self.assertFalse(self.kb.ask(('Unsafe', 'River_Rune')))


class TestPart2CNF(unittest.TestCase):
    """Part 2: (S ^ M) => G  becomes  ~S v ~M v G."""

    def test_negate_is_an_involution(self):
        self.assertEqual(negate('G'), '~G')
        self.assertEqual(negate('~G'), 'G')
        self.assertEqual(negate(negate('S')), 'S')

    def test_implication_to_cnf_clause(self):
        clause = to_cnf(['S', 'M'], 'G')
        self.assertEqual(clause, frozenset({'~S', '~M', 'G'}))

    def test_clause_str_is_a_disjunction(self):
        self.assertEqual(clause_str(to_cnf(['S', 'M'], 'G')), 'G v ~M v ~S')

    def test_single_premise_implication(self):
        # CanAttack => WinTeamFight  ==  ~CanAttack v WinTeamFight
        self.assertEqual(to_cnf(['CanAttack'], 'WinTeamFight'),
                         frozenset({'~CanAttack', 'WinTeamFight'}))


class TestPart3Resolution(unittest.TestCase):
    """Part 3: prove G from {~S v ~M v G, S, M} by contradiction."""

    def setUp(self):
        self.kb_clauses = [
            to_cnf(['S', 'M'], 'G'),
            frozenset({'S'}),
            frozenset({'M'}),
        ]

    def test_single_resolution_step(self):
        resolvents = resolve(frozenset({'~S', '~M', 'G'}), frozenset({'S'}))
        self.assertEqual(resolvents, [frozenset({'~M', 'G'})])

    def test_complementary_unit_clauses_give_the_empty_clause(self):
        self.assertIn(frozenset(), resolve(frozenset({'G'}), frozenset({'~G'})))

    def test_kb_entails_g(self):
        self.assertTrue(resolution_prove(self.kb_clauses, 'G'))

    def test_proof_trace_reaches_the_empty_clause(self):
        trace = []
        resolution_prove(self.kb_clauses, 'G', trace)
        self.assertTrue(any('Empty clause' in line for line in trace))
        self.assertTrue(any('negated goal' in line for line in trace))

    def test_missing_sensor_fact_blocks_the_proof(self):
        # Mana is not at 100%, so G is no longer entailed.
        without_mana = [to_cnf(['S', 'M'], 'G'), frozenset({'S'})]
        self.assertFalse(resolution_prove(without_mana, 'G'))

    def test_tautologies_are_discarded(self):
        # (A v B) resolved with (~A v ~B) would give (B v ~B), a tautology.
        self.assertEqual(resolve(frozenset({'A', 'B'}), frozenset({'~A', '~B'})), [])


class TestPart4Unification(unittest.TestCase):
    """Part 4: theta = {x/Spectre}."""

    VARS = {'x'}

    def test_the_tutorial_substitution(self):
        theta = unify(('Enemy', 'x'), ('Enemy', 'Spectre'), variables=self.VARS)
        self.assertEqual(theta, {'x': 'Spectre'})

    def test_theta_carries_across_both_premises(self):
        theta = unify(('Enemy', 'x'), ('Enemy', 'Spectre'), variables=self.VARS)
        theta = unify(('Invisible', 'x'), ('Invisible', 'Spectre'), theta,
                      variables=self.VARS)
        self.assertEqual(theta, {'x': 'Spectre'})

    def test_different_predicates_do_not_unify(self):
        self.assertIsNone(unify(('Enemy', 'x'), ('Invisible', 'Spectre'),
                                variables=self.VARS))

    def test_a_variable_cannot_take_two_values(self):
        theta = unify(('Enemy', 'x'), ('Enemy', 'Spectre'), variables=self.VARS)
        self.assertIsNone(unify(('Invisible', 'x'), ('Invisible', 'Riki'), theta,
                                variables=self.VARS))

    def test_rule_fires_only_on_a_consistent_binding(self):
        kb = KnowledgeBase(variables=self.VARS)
        kb.tell_rule([('Enemy', 'x'), ('Invisible', 'x')], ('CarryDust', 'Agent'))
        kb.tell_fact(('Enemy', 'Spectre'))
        kb.tell_fact(('Invisible', 'Riki'))     # a DIFFERENT hero is invisible
        self.assertFalse(kb.ask(('CarryDust', 'Agent')))
        kb.tell_fact(('Invisible', 'Spectre'))  # now one binding satisfies both
        self.assertTrue(kb.ask(('CarryDust', 'Agent')))


class TestPart5Chaining(unittest.TestCase):
    """Part 5: the same two Horn clauses, driven from both ends."""

    def build(self):
        kb = KnowledgeBase()
        kb.tell_rule([('HasDust',), ('TargetVisible',)], ('CanAttack',))
        kb.tell_rule([('CanAttack',)], ('WinTeamFight',))
        return kb

    def test_forward_chaining_derives_everything(self):
        kb = self.build()
        kb.tell_fact(('HasDust',))
        kb.tell_fact(('TargetVisible',))
        derived = kb.forward_chain()
        self.assertEqual(derived, {('CanAttack',), ('WinTeamFight',)})

    def test_forward_chaining_stops_at_a_fixed_point(self):
        kb = self.build()
        kb.tell_fact(('HasDust',))
        kb.tell_fact(('TargetVisible',))
        kb.forward_chain()
        self.assertEqual(kb.forward_chain(), set())  # nothing new the second time

    def test_forward_chaining_needs_both_premises(self):
        kb = self.build()
        kb.tell_fact(('HasDust',))              # TargetVisible missing
        self.assertEqual(kb.forward_chain(), set())

    def test_backward_chaining_proves_the_goal(self):
        kb = self.build()
        kb.tell_fact(('HasDust',))
        kb.tell_fact(('TargetVisible',))
        self.assertTrue(kb.backward_chain(('WinTeamFight',)))

    def test_backward_chaining_visits_goal_then_subgoals(self):
        kb = self.build()
        kb.tell_fact(('HasDust',))
        kb.tell_fact(('TargetVisible',))
        trace = []
        kb.backward_chain(('WinTeamFight',), trace)
        goals = [line.split('goal: ')[1] for line in trace if 'goal: ' in line]
        self.assertEqual(goals[0], 'WinTeamFight')      # starts from the goal
        self.assertIn('CanAttack', goals)               # then the sub-goal
        self.assertIn('HasDust', goals)                 # then the leaf facts

    def test_backward_chaining_fails_without_the_facts(self):
        self.assertFalse(self.build().backward_chain(('WinTeamFight',)))

    def test_backward_chaining_survives_a_cycle(self):
        kb = KnowledgeBase()
        kb.tell_rule([('A',)], ('B',))
        kb.tell_rule([('B',)], ('A',))
        self.assertFalse(kb.backward_chain(('A',)))     # terminates, does not hang


class TestSafetyKB(unittest.TestCase):
    """The grid-world KB the agent actually consults."""

    def test_unknown_cells_are_feasible(self):
        self.assertTrue(SafetyKB().is_feasible((4, 4)))

    def test_learned_toxic_cell_becomes_infeasible(self):
        kb = SafetyKB()
        kb.learn_toxic((2, 3))
        self.assertFalse(kb.is_feasible((2, 3)))
        self.assertTrue(kb.is_feasible((2, 4)))         # neighbours unaffected

    def test_learning_the_same_cell_twice_is_idempotent(self):
        kb = SafetyKB()
        self.assertTrue(kb.learn_toxic((1, 1)))
        self.assertFalse(kb.learn_toxic((1, 1)))

    def test_antidote_breaks_the_rule_premise(self):
        kb = SafetyKB()
        kb.learn_toxic((2, 3))
        self.assertFalse(kb.is_feasible((2, 3)))
        kb.take_antidote()
        self.assertTrue(kb.is_feasible((2, 3)))         # Toxic still true, Unsafe not entailed
        self.assertIn(('Toxic', cell((2, 3))), kb.facts)

    def test_known_toxic_positions_round_trips_coordinates(self):
        kb = SafetyKB()
        kb.learn_toxic((0, 0))
        kb.learn_toxic((11, 7))
        self.assertEqual(kb.known_toxic_positions(), {(0, 0), (11, 7)})


class TestLogicAgentIntegration(unittest.TestCase):
    """Search proposes, logic disposes -- on the actual grid agent."""

    CORRIDOR = (3, 1)

    def corridor_agent(self):
        agent = LogicAgent(active_algo='BFS', verbose=False)
        agent.pos, agent.facing = (0, 0), 'Right'
        return agent

    def test_search_agent_has_no_kb(self):
        self.assertFalse(hasattr(SearchAgent(), 'kb'))

    def test_path_exists_before_the_kb_learns_anything(self):
        agent = self.corridor_agent()
        self.assertEqual(agent._plan_to((2, 0), self.CORRIDOR, set()),
                         ['move_forward', 'move_forward'])

    def test_reachable_but_infeasible_after_telling_the_kb(self):
        agent = self.corridor_agent()
        agent.kb.learn_toxic((1, 0))
        agent._refresh_blocked()
        # The cell is still physically there -- no wall was added -- but the
        # planner now refuses to route through it, so no plan comes back.
        self.assertEqual(agent._plan_to((2, 0), self.CORRIDOR, set()), [])
        self.assertFalse(agent.kb.is_feasible((1, 0)))

    def test_antidote_makes_the_corridor_passable_again(self):
        agent = self.corridor_agent()
        agent.kb.learn_toxic((1, 0))
        agent._refresh_blocked()
        agent.kb.take_antidote()
        agent._refresh_blocked()
        self.assertEqual(agent._plan_to((2, 0), self.CORRIDOR, set()),
                         ['move_forward', 'move_forward'])

    def test_toxin_percept_is_told_to_the_kb(self):
        agent = self.corridor_agent()
        agent.pos = (1, 0)
        agent.sense_and_act({
            'smells_toxin': True, 'food_here': False, 'all_food': [(2, 0)],
            'grid_size': self.CORRIDOR, 'walls': [],
        })
        self.assertIn((1, 0), agent.kb.known_toxic_positions())
        self.assertEqual(agent.trap_hits, 1)

    def test_an_unsafe_plan_is_vetoed_and_replanned(self):
        agent = LogicAgent(active_algo='BFS', verbose=False)
        agent.pos, agent.facing = (0, 0), 'Right'
        agent.plan = ['move_forward', 'move_forward']   # heads through (1, 0)
        agent.kb.learn_toxic((1, 0))
        agent._refresh_blocked()
        self.assertFalse(agent._plan_is_feasible())

    def test_a_safe_plan_survives_validation(self):
        agent = LogicAgent(active_algo='BFS', verbose=False)
        agent.pos, agent.facing = (0, 0), 'Right'
        agent.plan = ['move_forward', 'move_forward']
        agent.kb.learn_toxic((5, 5))                    # irrelevant cell
        agent._refresh_blocked()
        self.assertTrue(agent._plan_is_feasible())

    def test_goal_selection_skips_infeasible_food(self):
        agent = LogicAgent(active_algo='BFS', verbose=False)
        agent.pos = (0, 0)
        agent.kb.learn_toxic((1, 0))                    # nearest food sits on a trap
        agent._refresh_blocked()
        self.assertEqual(agent._closest_food([(1, 0), (4, 0)]), (4, 0))

    def test_agent_never_freezes_when_every_route_is_blocked(self):
        # Food at (2,0) behind the only corridor cell, which is now unsafe.
        # The agent should proceed under protest rather than stall forever.
        agent = self.corridor_agent()
        agent.kb.learn_toxic((1, 0))
        agent._refresh_blocked()
        action = agent.sense_and_act({
            'smells_toxin': False, 'food_here': False, 'all_food': [(2, 0)],
            'grid_size': self.CORRIDOR, 'walls': [],
        })
        self.assertIsNotNone(action)
        self.assertNotEqual(agent.plan + [action], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
