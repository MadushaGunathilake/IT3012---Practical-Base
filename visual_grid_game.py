# visual_grid_game.py
import random
import tkinter as tk


class VisualGridHuntGame:
    """A flexible Pacman-style grid environment with support for configurable opponents and larger scales.

    The agent now has an orientation (facing direction) instead of moving in
    absolute compass directions directly. This is what makes partial
    observability meaningful: the agent can only sense the cell immediately
    in front of it (`wall_ahead`) and whether it is currently standing on
    food (`food_here`) -- it no longer gets its own global (x, y) coordinates.
    """

    # Clockwise ordering of facing directions, used for turning.
    TURN_ORDER = ['Up', 'Right', 'Down', 'Left']
    DIR_VECTORS = {
        'Up': (0, 1),
        'Right': (1, 0),
        'Down': (0, -1),
        'Left': (-1, 0),
    }

    def __init__(self, width=10, height=10, num_food=10, num_opponents=2, custom_walls=None):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]  # Starting position (x, y)
        self.agent_dir = 'Up'    # Starting facing direction

        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            # Generate some default scattered walls for a larger grid
            self.walls = {(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)}

        # Dynamically generate random food positions avoiding walls and agent start
        self.food_positions = set()
        while len(self.food_positions) < num_food:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            pos_tuple = (fx, fy)
            if pos_tuple != (0, 0) and pos_tuple not in self.walls:
                self.food_positions.add(pos_tuple)

        # Generate adversarial opponents
        self.opponents = []
        while len(self.opponents) < num_opponents:
            ox = random.randint(0, self.width - 1)
            oy = random.randint(0, self.height - 1)
            op_pos = [ox, oy]
            if tuple(op_pos) != (0, 0) and tuple(op_pos) not in self.walls and tuple(op_pos) not in self.food_positions:
                self.opponents.append(op_pos)

        self.score = 0
        self.steps = 0
        self.collision = False

    # ------------------------------------------------------------------
    # Helpers for orientation-based movement / sensing
    # ------------------------------------------------------------------
    def _turn(self, direction, way):
        """Return the new facing direction after turning 'left' or 'right'."""
        idx = self.TURN_ORDER.index(direction)
        if way == 'left':
            idx = (idx - 1) % 4
        else:
            idx = (idx + 1) % 4
        return self.TURN_ORDER[idx]

    def _cell_ahead(self):
        """The (x, y) cell immediately in front of the agent, given its facing direction."""
        dx, dy = self.DIR_VECTORS[self.agent_dir]
        return (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

    def _is_wall_or_out_of_bounds(self, cell):
        x, y = cell
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return cell in self.walls

    # ------------------------------------------------------------------
    # Step 1.1: Partial observability -- local booleans only, no global coords
    # ------------------------------------------------------------------
    def get_percept(self) -> dict:
        return {
            'wall_ahead': self._is_wall_or_out_of_bounds(self._cell_ahead()),
            'food_here': tuple(self.agent_pos) in self.food_positions,
            'collision': self.collision,
            'score': self.score,
            'remaining_food': len(self.food_positions),
        }

    def execute_action(self, action: str):
        """Actions are now relative to the agent's own orientation:
        'move_forward', 'turn_left', 'turn_right', 'suck'.
        """
        self.steps += 1

        if action == 'turn_left':
            self.agent_dir = self._turn(self.agent_dir, 'left')
        elif action == 'turn_right':
            self.agent_dir = self._turn(self.agent_dir, 'right')
        elif action == 'move_forward':
            ahead = self._cell_ahead()
            if self._is_wall_or_out_of_bounds(ahead):
                self.score -= 5
            else:
                self.agent_pos = list(ahead)
        elif action == 'suck':
            pos_tuple = tuple(self.agent_pos)
            if pos_tuple in self.food_positions:
                self.food_positions.remove(pos_tuple)
                self.score += 20
            else:
                self.score -= 1  # small penalty for sucking on an empty cell

        # Opponents still move randomly around the board each step
        for op in self.opponents:
            move = random.choice(['Up', 'Down', 'Left', 'Right', 'Stay'])
            if move == 'Up' and op[1] < self.height - 1:
                op[1] += 1
            elif move == 'Down' and op[1] > 0:
                op[1] -= 1
            elif move == 'Left' and op[0] > 0:
                op[0] -= 1
            elif move == 'Right' and op[0] < self.width - 1:
                op[0] += 1

            if op == self.agent_pos:
                self.score -= 50
                self.collision = True

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= 60 or self.collision


# ==========================================================================
# Step 1.2: The Simple Reflex Agent (no memory / no __init__ state)
# ==========================================================================
class SimpleReflexAgent:
    """Pure condition-action rules. No history is stored between calls, so this
    agent has no way of knowing it has already tried 'turn_left' in this exact
    spot before. In a corner or a U-shaped wall, this leads to it repeating the
    same cycle of actions forever (e.g. turn_left, move_forward, hit wall,
    turn_left, ...).
    """

    def sense_and_act(self, percept: dict) -> str:
        if percept['food_here']:
            return 'suck'
        elif percept['wall_ahead']:
            return 'turn_left'
        else:
            return 'move_forward'


# ==========================================================================
# Step 1.3: The Model-Based Agent (keeps an internal state/model of the world)
# ==========================================================================
class ModelBasedAgent:
    """Because the environment is only partially observable, this agent builds
    its own internal model of the world: an estimated relative position and
    facing direction (updated purely from its own actions -- it never sees
    global coordinates), plus a set of cells it believes it has already
    visited. That memory lets it recognize when moving forward or turning
    left would just send it back into visited territory, so it can choose a
    different way out instead of looping forever.
    """

    DIR_VECTORS = VisualGridHuntGame.DIR_VECTORS
    TURN_ORDER = VisualGridHuntGame.TURN_ORDER

    def __init__(self):
        self.pos_estimate = [0, 0]   # Relative position, agent's own frame of reference
        self.dir_estimate = 'Up'     # Relative facing direction
        self.visited_cells = {(0, 0)}
        self.last_action = None

    def _turn(self, direction, way):
        idx = self.TURN_ORDER.index(direction)
        idx = (idx - 1) % 4 if way == 'left' else (idx + 1) % 4
        return self.TURN_ORDER[idx]

    def _cell_in_direction(self, direction):
        dx, dy = self.DIR_VECTORS[direction]
        return (self.pos_estimate[0] + dx, self.pos_estimate[1] + dy)

    def _update_state(self, action):
        """Transition model: given the action just taken, update our belief
        about where we are and which way we're facing."""
        if action == 'move_forward':
            self.pos_estimate = list(self._cell_in_direction(self.dir_estimate))
            self.visited_cells.add(tuple(self.pos_estimate))
        elif action == 'turn_left':
            self.dir_estimate = self._turn(self.dir_estimate, 'left')
        elif action == 'turn_right':
            self.dir_estimate = self._turn(self.dir_estimate, 'right')
        # 'suck' changes neither position nor direction
        self.last_action = action

    def sense_and_act(self, percept: dict) -> str:
        # First, record what we currently know: we are here, at this point in time.
        self.visited_cells.add(tuple(self.pos_estimate))

        if percept['food_here']:
            action = 'suck'
        elif percept['wall_ahead']:
            # Always turn the SAME way (right) when blocked. This is a classic
            # wall-following trick: rotating consistently in one direction is
            # guaranteed to eventually find whichever opening let us into this
            # spot in the first place, instead of flip-flopping between two
            # headings forever (which is what happens if the turn direction
            # depends on memory that keeps changing turn to turn).
            action = 'turn_right'
        else:
            next_cell = self._cell_in_direction(self.dir_estimate)
            # The path ahead is clear. If we've already been to that cell,
            # turn away from it to go hunt for unexplored territory instead
            # of blindly re-walking the same path.
            action = 'turn_left' if next_cell in self.visited_cells else 'move_forward'

        self._update_state(action)
        return action


class GridGameGUI:
    """Tkinter wrapper that dynamically scales cell sizes to keep larger grids on screen."""

    AGENT_CLASSES = {
        'random': None,
        'simple_reflex': SimpleReflexAgent,
        'model_based': ModelBasedAgent,
    }

    def __init__(self, root, width=10, height=10, num_food=12, num_opponents=2, walls=None,
                 agent_type='simple_reflex'):
        self.root = root
        self.root.title("IT3012 - Scalable Multi-Agent Grid Hunt")

        self.env = VisualGridHuntGame(width=width, height=height, num_food=num_food, num_opponents=num_opponents,
                                      custom_walls=walls)

        self.agent_type = agent_type
        agent_cls = self.AGENT_CLASSES.get(agent_type)
        self.agent = agent_cls() if agent_cls else None

        # Dynamically calculate cell size so the total canvas fits nicely within a 600x600 window ceiling
        max_canvas_dim = 600
        self.cell_size = max(20, min(max_canvas_dim // self.env.width, max_canvas_dim // self.env.height))

        canvas_w = self.env.width * self.cell_size
        canvas_h = self.env.height * self.cell_size

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Score: 0 | Steps: 0", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn = tk.Button(root, text="Start Simulation", command=self.run_loop, font=("Arial", 12), bg="#000066",
                             fg="white")
        self.btn.pack(pady=5)

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")

        for x in range(self.env.width):
            for y in range(self.env.height):
                x1 = x * self.cell_size
                y1 = (self.env.height - 1 - y) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = "#f1f5f9" if (x, y) not in self.env.walls else "#64748b"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cbd5e1")

                # Only draw text if cell is large enough
                if self.cell_size >= 40 and (x, y) in self.env.walls:
                    self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2, text="W", fill="white",
                                            font=("Arial", 8, "bold"))

        for fx, fy in self.env.food_positions:
            offset = self.cell_size * 0.25
            x1 = fx * self.cell_size + offset
            y1 = (self.env.height - 1 - fy) * self.cell_size + offset
            self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.5, y1 + self.cell_size * 0.5, fill="#f59e0b",
                                    outline="#d97706")

        for ox, oy in self.env.opponents:
            offset = self.cell_size * 0.2
            x1 = ox * self.cell_size + offset
            y1 = (self.env.height - 1 - oy) * self.cell_size + offset
            self.canvas.create_rectangle(x1, y1, x1 + self.cell_size * 0.6, y1 + self.cell_size * 0.6, fill="#990000",
                                         outline="#7a0000")

        ax, ay = self.env.agent_pos
        offset = self.cell_size * 0.15
        x1 = ax * self.cell_size + offset
        y1 = (self.env.height - 1 - ay) * self.cell_size + offset
        x2 = x1 + self.cell_size * 0.7
        y2 = y1 + self.cell_size * 0.7
        self.canvas.create_oval(x1, y1, x2, y2, fill="#000066", outline="#1e3a8a")

        # Small line showing which way the agent is facing, since orientation
        # now matters (partial observability is direction-relative).
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = self.env.DIR_VECTORS[self.env.agent_dir]
        tip_x = cx + dx * self.cell_size * 0.4
        tip_y = cy - dy * self.cell_size * 0.4  # canvas y grows downward
        self.canvas.create_line(cx, cy, tip_x, tip_y, fill="#00e0ff", width=3)

    def run_loop(self):
        self.btn.config(state="disabled")

        def step():
            if not self.env.is_done():
                if self.agent is not None:
                    percept = self.env.get_percept()
                    action = self.agent.sense_and_act(percept)
                else:
                    action = random.choice(['turn_left', 'turn_right', 'move_forward'])

                self.env.execute_action(action)

                self.draw_grid()
                self.label.config(text=f"Score: {self.env.score} | Steps: {self.env.steps} | Action: {action}")
                self.root.after(250, step)
            else:
                end_text = f"Collision! Game Over! Final Score: {self.env.score}" if self.env.collision else f"Finished! Final Score: {self.env.score}"
                self.label.config(text=end_text)
                self.btn.config(state="normal")

        step()


if __name__ == "__main__":
    root = tk.Tk()
    # Try switching agent_type between 'simple_reflex' and 'model_based' to
    # compare Step 1.2's looping failure against Step 1.3's escape behavior.
    app = GridGameGUI(root, width=12, height=12, num_food=15, num_opponents=0, agent_type='simple_reflex')
    root.mainloop()