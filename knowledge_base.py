# knowledge_base.py
"""Tutorial 05 -- Logic Coverage.

Practicals 1-4 built the *Exploration* half of the agent: reflex rules, an
internal model, and BFS/DFS/UCS/A* search that answers "is this state
Reachable?".  This module builds the *Validation* half: a Knowledge Base and
an inference engine that answer "is this state Feasible?".

Everything the tutorial asks for on paper is implemented here as running code:

  Part 2  to_cnf()                -- implication elimination + De Morgan
  Part 3  resolution_prove()      -- proof by contradiction, empty clause
  Part 4  unify()                 -- FOL variable binding, substitution theta
  Part 5  forward_chain()         -- data-driven inference
          backward_chain()        -- goal-driven inference

Representation
--------------
A *literal* is a plain string.  A leading '~' means negation:  'G', '~S'.
A *clause* is a frozenset of literals, read as a disjunction:
    frozenset({'~S', '~M', 'G'})   ==   ~S v ~M v G
A CNF sentence is a list (conjunction) of such clauses.

For the First-Order Logic parts an *atom* is a tuple:
    ('Enemy', 'x')      ==  Enemy(x)
    ('Enemy', 'Spectre')==  Enemy(Spectre)
Anything in KnowledgeBase.variables (lowercase by default) is a variable.
"""

from collections import deque
from itertools import combinations


# ---------------------------------------------------------------------------
# Literal helpers
# ---------------------------------------------------------------------------

def negate(literal: str) -> str:
    """Return the complement of a literal.  negate('G') == '~G'."""
    return literal[1:] if literal.startswith('~') else '~' + literal


def clause_str(clause) -> str:
    """Human-readable form of a clause, e.g. '~S v ~M v G'.  {} is the empty
    clause, the contradiction that ends a resolution proof."""
    if not clause:
        return '{}  (empty clause / contradiction)'
    return ' v '.join(sorted(clause, key=lambda lit: lit.lstrip('~')))


# ---------------------------------------------------------------------------
# Part 2 -- CNF conversion
# ---------------------------------------------------------------------------

def to_cnf(premises, conclusion):
    """Tutorial Part 2, Steps A and B.

    Takes an implication  (p1 ^ p2 ^ ... ^ pn) => c  and returns the single
    equivalent CNF clause  ~p1 v ~p2 v ... v ~pn v c.

    The two textbook steps, applied to (S ^ M) => G:

      Step 1 -- Implication Elimination,  a => b  ==  ~a v b
                (S ^ M) => G   ==   ~(S ^ M) v G
      Step 2 -- De Morgan's Law,  ~(a ^ b)  ==  ~a v ~b
                ~(S ^ M) v G   ==   ~S v ~M v G

    >>> clause_str(to_cnf(['S', 'M'], 'G'))
    '~M v ~S v G'
    """
    return frozenset(negate(p) for p in premises) | {conclusion}


def cnf_derivation(premises, conclusion):
    """Return the worked steps of to_cnf() as strings, so the agent can print
    the same derivation the tutorial asks you to write out by hand."""
    antecedent = ' ^ '.join(premises)
    return [
        f'Original:                ({antecedent}) => {conclusion}',
        f'Implication Elimination: ~({antecedent}) v {conclusion}',
        f"De Morgan's Law:         "
        + ' v '.join([negate(p) for p in premises] + [conclusion]),
    ]


# ---------------------------------------------------------------------------
# Part 3 -- Proof by Resolution
# ---------------------------------------------------------------------------

def resolve(clause_a, clause_b):
    """Apply the Resolution rule to two clauses.

    If clause_a contains a literal whose complement is in clause_b, the two
    clauses "smash together": the complementary pair cancels and everything
    else is unioned into a new clause (the resolvent).

        (~S v ~M v G)  and  (S)      ->   (~M v G)

    Returns the list of resolvents (one per complementary pair).  A resolvent
    that is the empty frozenset is the contradiction we are hunting for.
    """
    resolvents = []
    for literal in clause_a:
        if negate(literal) in clause_b:
            resolvent = (clause_a - {literal}) | (clause_b - {negate(literal)})
            # A clause containing both P and ~P is a tautology (always true)
            # and is useless in a proof, so drop it.
            if any(negate(lit) in resolvent for lit in resolvent):
                continue
            resolvents.append(frozenset(resolvent))
    return resolvents


def resolution_prove(kb_clauses, query, trace=None):
    """Tutorial Part 3: prove `query` from `kb_clauses` by contradiction.

    Method: assume the opposite.  Add ~query to the KB and resolve everything
    against everything.  If we ever derive the empty clause the assumption is
    impossible, so the query must be entailed.

    kb_clauses -- iterable of frozensets (CNF)
    query      -- a single literal, e.g. 'G'
    trace      -- optional list; each proof step is appended as a string

    Returns True if KB entails query, otherwise False.
    """
    clauses = {frozenset(c) for c in kb_clauses}
    negated = frozenset({negate(query)})

    if trace is not None:
        for i, clause in enumerate(sorted(clauses, key=len), start=1):
            trace.append(f'  {i}. {clause_str(clause)}   (KB)')
        trace.append(f'  +. {clause_str(negated)}   (negated goal, assumed for contradiction)')

    clauses.add(negated)

    while True:
        new = set()
        for clause_a, clause_b in combinations(clauses, 2):
            for resolvent in resolve(clause_a, clause_b):
                if trace is not None and resolvent not in clauses and resolvent not in new:
                    trace.append(
                        f'  resolve [{clause_str(clause_a)}] with [{clause_str(clause_b)}]'
                        f'  ->  {clause_str(resolvent)}'
                    )
                if not resolvent:
                    if trace is not None:
                        trace.append(f'  Empty clause reached: KB entails {query}.')
                    return True
                new.add(resolvent)

        if new.issubset(clauses):
            # Saturated: no new clauses can be derived and no contradiction.
            if trace is not None:
                trace.append(f'  No new resolvents. KB does NOT entail {query}.')
            return False
        clauses |= new


# ---------------------------------------------------------------------------
# Part 4 -- First-Order Logic & Unification
# ---------------------------------------------------------------------------

def is_variable(term, variables=None):
    """A term is a variable if it is listed in `variables`, or (by default
    convention) if it is a lowercase string like 'x'."""
    if not isinstance(term, str):
        return False
    if variables is not None:
        return term in variables
    return term.islower()


def unify(pattern, fact, theta=None, variables=None):
    """Tutorial Part 4: find the substitution theta that makes two atoms
    identical, or return None if no such substitution exists.

    >>> unify(('Enemy', 'x'), ('Enemy', 'Spectre'))
    {'x': 'Spectre'}
    >>> unify(('Enemy', 'x'), ('Invisible', 'Spectre')) is None
    True

    theta is threaded through successive calls so that one variable stays
    bound to one constant across every premise of a rule -- this is what stops
    Enemy(Spectre) ^ Invisible(Riki) from firing the rule.
    """
    if theta is None:
        theta = {}

    if len(pattern) != len(fact):
        return None

    for p_term, f_term in zip(pattern, fact):
        if is_variable(p_term, variables):
            bound = theta.get(p_term)
            if bound is None:
                theta = dict(theta)
                theta[p_term] = f_term
            elif bound != f_term:
                return None  # variable already bound to a different constant
        elif p_term != f_term:
            return None      # two constants that simply don't match
    return theta


def substitute(atom, theta):
    """Apply substitution theta to an atom.  substitute(('Enemy','x'),
    {'x': 'Spectre'}) -> ('Enemy', 'Spectre')."""
    return tuple(theta.get(term, term) for term in atom)


# ---------------------------------------------------------------------------
# Part 5 -- Horn clauses, Forward and Backward chaining
# ---------------------------------------------------------------------------

class Rule:
    """A definite (Horn) clause:  premises => conclusion.

    Premises and conclusion are atoms (tuples).  Propositional rules are just
    1-element tuples: ('HasDust',) ^ ('TargetVisible',) => ('CanAttack',).
    """

    def __init__(self, premises, conclusion):
        self.premises = [tuple(p) for p in premises]
        self.conclusion = tuple(conclusion)

    def __repr__(self):
        prem = ' ^ '.join(atom_str(p) for p in self.premises)
        return f'{prem} => {atom_str(self.conclusion)}'


def atom_str(atom):
    """('Enemy', 'Spectre') -> 'Enemy(Spectre)';  ('HasDust',) -> 'HasDust'."""
    if len(atom) == 1:
        return atom[0]
    return f'{atom[0]}({", ".join(atom[1:])})'


class KnowledgeBase:
    """The agent's Knowledge Base plus its inference engine.

    Holds ground facts (atoms known true) and Horn rules, and can be queried
    data-driven (forward_chain) or goal-driven (backward_chain).
    """

    def __init__(self, variables=None):
        self.facts = set()
        self.rules = []
        self.variables = set(variables) if variables else None

    # -- TELL ---------------------------------------------------------------

    def tell_fact(self, atom):
        """Assert a ground fact, e.g. tell_fact(('Toxic', '3,4'))."""
        self.facts.add(tuple(atom))
        return self

    def tell_rule(self, premises, conclusion):
        """Assert a Horn rule, e.g.
        tell_rule([('Enemy', 'x'), ('Invisible', 'x')], ('CarryDust', 'Agent'))"""
        self.rules.append(Rule(premises, conclusion))
        return self

    def retract(self, atom):
        """Remove a fact.  Needed because the world changes -- placing a Ward
        makes DarkMap false, which un-fires the 'River is Unsafe' rule."""
        self.facts.discard(tuple(atom))
        return self

    # -- ASK ----------------------------------------------------------------

    def _matching_substitutions(self, premises, theta):
        """Yield every substitution extending theta that satisfies all
        premises against the current facts.  This is where unify() does the
        FOL variable binding from Part 4."""
        if not premises:
            yield theta
            return

        head, rest = premises[0], premises[1:]
        for fact in self.facts:
            new_theta = unify(head, fact, theta, self.variables)
            if new_theta is not None:
                yield from self._matching_substitutions(rest, new_theta)

    def forward_chain(self, trace=None):
        """Part 5, Trace 1 -- DATA-DRIVEN inference.

        Start from what the sensors gave us and repeatedly fire every rule
        whose premises are already satisfied, adding the conclusions back into
        the fact set.  Repeat until nothing new can be derived (a fixed
        point).  It computes *everything* that follows, whether or not anyone
        asked -- that is both its strength (it never misses a consequence) and
        its cost (it derives irrelevant facts too).

        Returns the set of newly derived facts.
        """
        derived = set()
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                # Materialise the matches before mutating self.facts, otherwise
                # we would be adding to the set the generator is iterating over.
                for theta in list(self._matching_substitutions(rule.premises, {})):
                    conclusion = substitute(rule.conclusion, theta)
                    if conclusion not in self.facts:
                        self.facts.add(conclusion)
                        derived.add(conclusion)
                        changed = True
                        if trace is not None:
                            bound = ', '.join(f'{k}/{v}' for k, v in theta.items()) or '{}'
                            trace.append(
                                f'  fire [{rule}] with theta = {{{bound}}}'
                                f'  ->  derive {atom_str(conclusion)}'
                            )
        return derived

    def backward_chain(self, goal, trace=None, depth=0, seen=None):
        """Part 5, Trace 2 -- GOAL-DRIVEN inference.

        Start from the goal and work backwards. If the goal is already a known
        fact, done. Otherwise find rules that *conclude* the goal and
        recursively try to prove each of their premises as sub-goals. It only
        ever touches facts relevant to the question asked, so it does far less
        work than forward chaining -- but it must be asked a question first.

        Premises may contain unbound variables (the rule
        Enemy(x) ^ Invisible(x) => CarryDust(Agent) has none in its
        conclusion), so proving a conjunction means searching for ONE
        substitution that satisfies every premise at once. That is what
        _prove_conjunction does, and it is why theta is threaded through.

        Returns True if the goal is provable.
        """
        goal = tuple(goal)
        if trace is not None:
            trace.append('  ' + '  ' * depth + f'goal: {atom_str(goal)}')

        for _theta in self._prove(goal, {}, trace, depth, frozenset()):
            return True

        if trace is not None:
            trace.append('  ' + '  ' * depth + '  not provable -> FALSE')
        return False

    def _prove(self, goal, theta, trace, depth, seen):
        """Yield every substitution under which `goal` is provable."""
        goal = substitute(tuple(goal), theta)
        pad = '  ' + '  ' * depth

        # Base case: the goal matches something we already know.
        for fact in self.facts:
            new_theta = unify(goal, fact, theta, self.variables)
            if new_theta is not None:
                if trace is not None:
                    trace.append(f'{pad}  known fact {atom_str(fact)} -> TRUE')
                yield new_theta

        if goal in seen:            # cycle guard, e.g. A => B and B => A
            if trace is not None:
                trace.append(f'{pad}  already being proved (cycle) -> stop')
            return

        # Recursive case: any rule that concludes this goal.
        for rule in self.rules:
            rule_theta = unify(rule.conclusion, goal, theta, self.variables)
            if rule_theta is None:
                continue
            if trace is not None:
                trace.append(f'{pad}  rule [{rule}] concludes it; prove its premises:')
            for result in self._prove_conjunction(
                rule.premises, rule_theta, trace, depth + 2, seen | {goal}
            ):
                if trace is not None:
                    trace.append(f'{pad}  all premises hold -> {atom_str(goal)} is TRUE')
                yield result

    def _prove_conjunction(self, premises, theta, trace, depth, seen):
        """Yield every substitution that proves ALL premises together. One
        variable keeps one binding across the whole conjunction, which is what
        stops Enemy(Spectre) ^ Invisible(Riki) from firing the rule."""
        if not premises:
            yield theta
            return

        head, rest = premises[0], premises[1:]
        if trace is not None:
            trace.append('  ' + '  ' * depth + f'goal: {atom_str(substitute(head, theta))}')
        for new_theta in self._prove(head, theta, trace, depth, seen):
            yield from self._prove_conjunction(rest, new_theta, trace, depth, seen)

    def ask(self, goal):
        """Convenience wrapper: is `goal` entailed?  Uses backward chaining
        because it is a specific question about one state."""
        return self.backward_chain(goal)


# ---------------------------------------------------------------------------
# Part 1 -- Reachability vs Feasibility, applied to the grid world
# ---------------------------------------------------------------------------

def cell(pos) -> str:
    """Encode a grid coordinate as a constant symbol usable inside an atom.
    (3, 4) -> 'C3_4'.  Uppercase so is_variable() never mistakes it for one."""
    return f'C{pos[0]}_{pos[1]}'


class SafetyKB(KnowledgeBase):
    """The grid-world Knowledge Base the SearchAgent consults before it
    commits to an A*/BFS plan.

    Holds the FOL safety rule from Tutorial Part 4, restated for our map:

        forall x  Toxic(x) ^ NoAntidote(Agent)  =>  Unsafe(x)

    which is structurally identical to the tutorial's

        forall x  Enemy(x) ^ Invisible(x)  =>  CarryDust(Agent)

    The environment never hands out the trap positions (get_percept only
    exposes the local boolean 'smells_toxin'), so Toxic(x) facts are *learned*
    at runtime: every time the agent stands on a cell and the toxin sensor
    fires, it TELLs the KB.  From then on that cell stays reachable but
    becomes infeasible, and the planner must route around it.
    """

    AGENT = 'Agent'

    def __init__(self):
        super().__init__(variables={'x'})
        # Standing rule: a toxic cell is unsafe unless we carry an antidote.
        self.tell_rule([('Toxic', 'x'), ('NoAntidote', self.AGENT)], ('Unsafe', 'x'))
        # Current sensor state: we have no antidote, so the rule is armed.
        self.tell_fact(('NoAntidote', self.AGENT))

    def learn_toxic(self, pos) -> bool:
        """Sensor -> KB.  Returns True if this is newly learned."""
        atom = ('Toxic', cell(pos))
        if atom in self.facts:
            return False
        self.tell_fact(atom)
        return True

    def take_antidote(self):
        """Falsify the rule's second premise.  Exactly the tutorial's advice
        for the River: to make an unsafe state feasible, break a premise
        rather than argue with the conclusion."""
        self.retract(('NoAntidote', self.AGENT))
        return self

    def is_feasible(self, pos) -> bool:
        """Feasibility check: a cell is feasible unless the KB entails
        Unsafe(cell).  Note this is *not* a search question -- search already
        said the cell is reachable."""
        return not self.backward_chain(('Unsafe', cell(pos)))

    def unsafe_cells(self, candidate_positions):
        """Every candidate position the KB currently rules out.  Handed to the
        planner as extra 'walls' so search never proposes an infeasible path."""
        return {pos for pos in candidate_positions if not self.is_feasible(pos)}

    def known_toxic_positions(self):
        """Decode the learned Toxic(x) facts back into (x, y) tuples."""
        out = set()
        for atom in self.facts:
            if atom[0] == 'Toxic':
                x, y = atom[1][1:].split('_')
                out.add((int(x), int(y)))
        return out


# ---------------------------------------------------------------------------
# Tutorial demonstrations -- run this file directly to see every part work
# ---------------------------------------------------------------------------

def demo_part_1():
    print('=' * 72)
    print('Part 1 -- Reachability vs Feasibility (Mid_Tower -> River_Rune)')
    print('=' * 72)

    kb = KnowledgeBase()
    kb.tell_rule([('DarkMap',), ('BloodseekerMissing',)], ('Unsafe', 'River_Rune'))
    kb.tell_fact(('DarkMap',))
    kb.tell_fact(('BloodseekerMissing',))

    trace = []
    unsafe = kb.backward_chain(('Unsafe', 'River_Rune'), trace)
    print('\n'.join(trace))
    print(f'\n  A* found a path            -> River_Rune is REACHABLE: True')
    print(f'  KB entails Unsafe(River)   -> River_Rune is FEASIBLE:  {not unsafe}')
    print('  Reachable but infeasible: search proposes, logic vetoes.\n')

    kb.retract(('DarkMap',))          # place a Ward -> premise falsified
    still_unsafe = kb.ask(('Unsafe', 'River_Rune'))
    print(f'  After placing a Ward (DarkMap retracted) -> FEASIBLE: {not still_unsafe}\n')


def demo_part_2_and_3():
    print('=' * 72)
    print('Part 2 -- CNF conversion   /   Part 3 -- Proof by Resolution')
    print('=' * 72)

    for line in cnf_derivation(['S', 'M'], 'G'):
        print('  ' + line)

    kb_clauses = [
        to_cnf(['S', 'M'], 'G'),   # ~S v ~M v G
        frozenset({'S'}),          # sensor fact: ShadowBlade in inventory
        frozenset({'M'}),          # sensor fact: Mana at 100%
    ]
    trace = []
    proved = resolution_prove(kb_clauses, 'G', trace)
    print('\n  Proof by contradiction of G:')
    print('\n'.join(trace))
    print(f'\n  Result: G is entailed = {proved}  -> Drow CAN safely Gank.\n')


def demo_part_4():
    print('=' * 72)
    print('Part 4 -- First-Order Logic & Unification')
    print('=' * 72)

    kb = KnowledgeBase(variables={'x'})
    kb.tell_rule([('Enemy', 'x'), ('Invisible', 'x')], ('CarryDust', 'Agent'))
    kb.tell_fact(('Enemy', 'Spectre'))
    kb.tell_fact(('Invisible', 'Spectre'))

    theta = unify(('Enemy', 'x'), ('Enemy', 'Spectre'), variables={'x'})
    print(f'  unify(Enemy(x), Enemy(Spectre))       -> theta = {theta}')
    theta = unify(('Invisible', 'x'), ('Invisible', 'Spectre'), theta, variables={'x'})
    print(f'  unify(Invisible(x), Invisible(Spectre)) -> theta = {theta}')
    print('  One consistent binding x/Spectre satisfies BOTH premises,')
    print('  so the rule fires and the agent concludes CarryDust(Agent).')

    trace = []
    kb.forward_chain(trace)
    print('\n'.join(trace))

    bad = unify(('Enemy', 'x'), ('Invisible', 'Spectre'), variables={'x'})
    print(f'\n  unify(Enemy(x), Invisible(Spectre))   -> {bad}  (predicates differ)\n')


def demo_part_5():
    print('=' * 72)
    print('Part 5 -- Forward vs Backward Chaining')
    print('=' * 72)

    def fresh():
        kb = KnowledgeBase()
        kb.tell_rule([('HasDust',), ('TargetVisible',)], ('CanAttack',))
        kb.tell_rule([('CanAttack',)], ('WinTeamFight',))
        return kb

    print('\n  Trace 1 -- DATA-DRIVEN (forward chaining), sensors fire first:')
    kb = fresh()
    kb.tell_fact(('HasDust',))
    kb.tell_fact(('TargetVisible',))
    trace = []
    derived = kb.forward_chain(trace)
    print('\n'.join(trace))
    print(f'  Derived without being asked: {sorted(atom_str(a) for a in derived)}')

    print('\n  Trace 2 -- GOAL-DRIVEN (backward chaining), goal given first:')
    kb = fresh()
    kb.tell_fact(('HasDust',))
    kb.tell_fact(('TargetVisible',))
    trace = []
    result = kb.backward_chain(('WinTeamFight',), trace)
    print('\n'.join(trace))
    print(f'  WinTeamFight provable = {result}\n')


if __name__ == '__main__':
    demo_part_1()
    demo_part_2_and_3()
    demo_part_4()
    demo_part_5()
