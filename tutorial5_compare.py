# tutorial5_compare.py
"""Tutorial 05 -- headless proof that logical validation is worth the compute.

Runs the SAME map (same seed -> same walls, food and traps) twice:

  SearchAgent  -- exploration only. Asks "is it reachable?" and nothing else,
                  so it walks over toxic traps repeatedly, paying -15 each time.
  LogicAgent   -- exploration + validation. The first time a trap fires the
                  toxin sensor it TELLs the KB Toxic(cell), ASKs whether
                  Unsafe(cell) is entailed, and re-plans around it. It can be
                  fooled once per trap; it is never fooled twice.

Run:  python tutorial5_compare.py
"""

import random

from agent import LogicAgent, SearchAgent
from visual_grid_game import VisualGridHuntGame

SEED = 7
MAX_STEPS = 400


def run(agent, seed=SEED, label=''):
    random.seed(seed)
    env = VisualGridHuntGame(width=12, height=12, num_food=10,
                             num_opponents=0, num_traps=14)

    trap_entries = 0
    steps = 0
    while env.food_positions and steps < MAX_STEPS:
        percept = env.get_percept()
        if percept['smells_toxin']:
            trap_entries += 1
        action = agent.sense_and_act(percept)
        env.execute_action(action)
        steps += 1

    return {
        'label': label,
        'score': env.score,
        'steps': steps,
        'trap_entries': trap_entries,
        'food_left': len(env.food_positions),
        'vetoes': getattr(agent, 'vetoed_plans', 0),
        'kb_facts': len(getattr(agent, 'kb', None).known_toxic_positions())
                    if hasattr(agent, 'kb') else 0,
    }


def main():
    print('Tutorial 05 -- Exploration alone vs Exploration + Logical Validation')
    print('Identical map (seed={}), 14 toxic traps, 10 food pellets.\n'.format(SEED))

    results = [
        run(SearchAgent(active_algo='AStar', verbose=False), label='SearchAgent  (search only)'),
        run(LogicAgent(active_algo='AStar', verbose=False),
            label='LogicAgent   (search + KB)'),
    ]

    header = '{:<28} {:>7} {:>7} {:>13} {:>10} {:>8} {:>8}'.format(
        'Agent', 'Score', 'Steps', 'Trap entries', 'Food left', 'Vetoes', 'KB facts')
    print(header)
    print('-' * len(header))
    for r in results:
        print('{label:<28} {score:>7} {steps:>7} {trap_entries:>13} '
              '{food_left:>10} {vetoes:>8} {kb_facts:>8}'.format(**r))

    delta = results[1]['score'] - results[0]['score']
    print('\nScore difference from adding the reasoning layer: {:+d}'.format(delta))
    print('Every trap cell stayed REACHABLE for both agents -- A* could always')
    print('find a path through one. Only the KB made those cells INFEASIBLE.')


if __name__ == '__main__':
    main()
