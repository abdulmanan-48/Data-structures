import pygame
import heapq
import math
import time
import sys

# ─── Constants ────────────────────────────────────────────────────────────────
ROWS, COLS = 8, 8
CELL = 80          # pixels per cell
PANEL_W = 280      # right-side dashboard width
WIN_W = COLS * CELL + PANEL_W
WIN_H = ROWS * CELL

FPS = 30
STEP_DELAY = 120   # ms between animation steps

# Colours
WHITE   = (255, 255, 255)
BLACK   = (30,  30,  30)
GRAY    = (180, 180, 180)
DKGRAY  = (60,  60,  60)
START_C = (50,  200, 80)    # green
GOAL_C  = (220, 50,  50)    # red
WALL_C  = (40,  40,  40)
FRONT_C = (255, 220, 0)     # yellow  – frontier
VISIT_C = (100, 149, 237)   # blue    – visited
PATH_C  = (50,  220, 130)   # green   – final path
PANEL_C = (25,  25,  40)
BTN_C   = (60,  80,  140)
BTN_H   = (90,  120, 200)
BTN_ACT = (40,  180, 100)
TEXT_C  = (230, 230, 230)
TITLE_C = (180, 210, 255)

# Cell type IDs
EMPTY, WALL, START, GOAL = 0, 1, 2, 3

# ─── Grid definition (same layout as Q7.ipynb) ────────────────────────────────
def build_grid():
    grid = [[EMPTY]*COLS for _ in range(ROWS)]
    # vertical walls
    for i in range(2, 6):
        grid[i][3] = WALL
        grid[i][5] = WALL
    # horizontal walls
    for j in range(1, 4):
        grid[4][j] = WALL
    for j in range(4, 7):
        grid[6][j] = WALL
    grid[1][1] = START
    grid[6][6] = GOAL
    return grid

START_POS = (1, 1)
GOAL_POS  = (6, 6)

MOVES = [(-1,0),(1,0),(0,-1),(0,1)]   # 4-directional

def neighbors(pos, grid):
    r, c = pos
    result = []
    for dr, dc in MOVES:
        nr, nc = r+dr, c+dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] != WALL:
            result.append((nr, nc))
    return result

# ─── Heuristics ───────────────────────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ─── Search algorithms (return generator for step-by-step animation) ──────────
def gbfs(grid, start, goal, heuristic):
    """Greedy Best-First Search – f(n) = h(n)"""
    h = heuristic
    open_set = [(h(start, goal), 0, start)]   # tie-break with counter
    counter = 1
    came_from = {start: None}
    visited = set()
    nodes_visited = 0
    start_time = time.time()

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        nodes_visited += 1
        yield ('visit', current, visited.copy(), set(n for _,_,n in open_set), None, nodes_visited, 0, 0)

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            cost = len(path) - 1
            elapsed = (time.time() - start_time) * 1000
            yield ('done', None, visited, set(), path, nodes_visited, cost, elapsed)
            return

        for nb in neighbors(current, grid):
            if nb not in visited and nb not in came_from:
                came_from[nb] = current
                heapq.heappush(open_set, (h(nb, goal), counter, nb))
                counter += 1
        yield ('frontier', current, visited.copy(), set(n for _,_,n in open_set), None, nodes_visited, 0, 0)

    yield ('done', None, visited, set(), None, nodes_visited, 0, (time.time()-start_time)*1000)


def astar(grid, start, goal, heuristic):
    """A* Search – f(n) = g(n) + h(n)"""
    h = heuristic
    open_set = [(h(start, goal), 0, start)]
    counter = 1
    came_from = {start: None}
    g_score = {start: 0}
    visited = set()
    nodes_visited = 0
    start_time = time.time()

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        nodes_visited += 1
        yield ('visit', current, visited.copy(), set(n for _,_,n in open_set), None, nodes_visited, 0, 0)

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            cost = g_score[goal]
            elapsed = (time.time() - start_time) * 1000
            yield ('done', None, visited, set(), path, nodes_visited, round(cost,2), elapsed)
            return

        for nb in neighbors(current, grid):
            tentative_g = g_score[current] + 1
            if nb not in g_score or tentative_g < g_score[nb]:
                g_score[nb] = tentative_g
                f = tentative_g + h(nb, goal)
                came_from[nb] = current
                heapq.heappush(open_set, (f, counter, nb))
                counter += 1
        yield ('frontier', current, visited.copy(), set(n for _,_,n in open_set), None, nodes_visited, 0, 0)

    yield ('done', None, visited, set(), None, nodes_visited, 0, (time.time()-start_time)*1000)


# ─── UI helpers ───────────────────────────────────────────────────────────────
def draw_button(surf, rect, text, font, active=False, hover=False):
    colour = BTN_ACT if active else (BTN_H if hover else BTN_C)
    pygame.draw.rect(surf, colour, rect, border_radius=8)
    pygame.draw.rect(surf, GRAY, rect, 1, border_radius=8)
    lbl = font.render(text, True, WHITE)
    surf.blit(lbl, lbl.get_rect(center=rect.center))

def draw_grid(surf, grid, visited, frontier, path):
    for r in range(ROWS):
        for c in range(COLS):
            x, y = c*CELL, r*CELL
            cell_type = grid[r][c]

            if cell_type == WALL:
                colour = WALL_C
            elif cell_type == START:
                colour = START_C
            elif cell_type == GOAL:
                colour = GOAL_C
            elif path and (r,c) in path:
                colour = PATH_C
            elif (r,c) in visited:
                colour = VISIT_C
            elif (r,c) in frontier:
                colour = FRONT_C
            else:
                colour = WHITE

            pygame.draw.rect(surf, colour, (x+1, y+1, CELL-2, CELL-2), border_radius=4)

    # grid lines
    for r in range(ROWS+1):
        pygame.draw.line(surf, GRAY, (0, r*CELL), (COLS*CELL, r*CELL))
    for c in range(COLS+1):
        pygame.draw.line(surf, GRAY, (c*CELL, 0), (c*CELL, ROWS*CELL))

def draw_panel(surf, font_big, font_med, font_sm,
               algo, heur, nodes, cost, elapsed,
               btn_rects, hover_idx, algo_idx, heur_idx,
               state):
    px = COLS * CELL
    pygame.draw.rect(surf, PANEL_C, (px, 0, PANEL_W, WIN_H))
    pygame.draw.line(surf, GRAY, (px, 0), (px, WIN_H), 2)

    # Title
    t = font_big.render("Search Visualizer", True, TITLE_C)
    surf.blit(t, (px+10, 10))

    y = 50
    surf.blit(font_med.render("── Algorithm ──", True, GRAY), (px+10, y)); y+=24
    for i,(lbl,r) in enumerate(zip(["GBFS","A*"], btn_rects['algo'])):
        draw_button(surf, r, lbl, font_med, active=(i==algo_idx), hover=(hover_idx==('algo',i)))
    y = btn_rects['algo'][0].bottom + 10

    surf.blit(font_med.render("── Heuristic ──", True, GRAY), (px+10, y)); y+=24
    for i,(lbl,r) in enumerate(zip(["Manhattan","Euclidean"], btn_rects['heur'])):
        draw_button(surf, r, lbl, font_med, active=(i==heur_idx), hover=(hover_idx==('heur',i)))
    y = btn_rects['heur'][0].bottom + 14

    # Run / Reset buttons
    draw_button(surf, btn_rects['run'], "▶  Run", font_med,
                hover=(hover_idx==('ctrl','run')))
    draw_button(surf, btn_rects['reset'], "↺  Reset", font_med,
                hover=(hover_idx==('ctrl','reset')))
    y = btn_rects['run'].bottom + 20

    # Metrics
    surf.blit(font_med.render("── Metrics ──", True, GRAY), (px+10, y)); y+=26
    metrics = [
        ("Nodes Visited", str(nodes)),
        ("Path Cost",     str(cost) if cost else "—"),
        ("Time (ms)",     f"{elapsed:.1f}" if elapsed else "—"),
    ]
    for label, val in metrics:
        surf.blit(font_sm.render(label+":", True, GRAY),  (px+14, y))
        surf.blit(font_sm.render(val, True, TEXT_C), (px+170, y))
        y += 22

    y += 10
    # Legend
    surf.blit(font_med.render("── Legend ──", True, GRAY), (px+10, y)); y+=26
    legend = [
        (START_C, "Start"),
        (GOAL_C,  "Goal"),
        (WALL_C,  "Wall"),
        (FRONT_C, "Frontier"),
        (VISIT_C, "Visited"),
        (PATH_C,  "Final Path"),
    ]
    for colour, lbl in legend:
        pygame.draw.rect(surf, colour, (px+14, y+3, 16, 16), border_radius=3)
        surf.blit(font_sm.render(lbl, True, TEXT_C), (px+36, y))
        y += 22

    y += 10
    status_map = {'idle':'Idle','running':'Running…','done':'Done ✓','no_path':'No path found'}
    s = status_map.get(state, state)
    surf.blit(font_med.render("Status: "+s, True, TITLE_C), (px+10, y))


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    surf = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("GBFS / A* Search Visualizer")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("segoeui", 18, bold=True)
    font_med = pygame.font.SysFont("segoeui", 15, bold=True)
    font_sm  = pygame.font.SysFont("segoeui", 14)

    px = COLS * CELL
    # Button layout
    btn_w, btn_h = 110, 32
    bx = px + 14
    btn_rects = {
        'algo': [
            pygame.Rect(bx,        74, btn_w, btn_h),
            pygame.Rect(bx+btn_w+8,74, btn_w, btn_h),
        ],
        'heur': [
            pygame.Rect(bx,        148, btn_w, btn_h),
            pygame.Rect(bx+btn_w+8,148, btn_w, btn_h),
        ],
        'run':   pygame.Rect(bx,        192, btn_w, btn_h),
        'reset': pygame.Rect(bx+btn_w+8,192, btn_w, btn_h),
    }

    # State
    grid = build_grid()
    algo_idx = 0   # 0=GBFS, 1=A*
    heur_idx = 0   # 0=Manhattan, 1=Euclidean
    visited  = set()
    frontier = set()
    path     = []
    nodes_visited = 0
    path_cost = 0
    elapsed   = 0.0
    state     = 'idle'   # idle | running | done | no_path
    gen       = None
    last_step = 0

    hover_idx = None

    running = True
    while running:
        now = pygame.time.get_ticks()
        mx, my = pygame.mouse.get_pos()

        # Detect hover
        hover_idx = None
        for i, r in enumerate(btn_rects['algo']):
            if r.collidepoint(mx, my): hover_idx = ('algo', i)
        for i, r in enumerate(btn_rects['heur']):
            if r.collidepoint(mx, my): hover_idx = ('heur', i)
        if btn_rects['run'].collidepoint(mx, my):   hover_idx = ('ctrl','run')
        if btn_rects['reset'].collidepoint(mx, my): hover_idx = ('ctrl','reset')

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, r in enumerate(btn_rects['algo']):
                    if r.collidepoint(mx, my):
                        algo_idx = i
                        state = 'idle'; gen = None
                        visited = set(); frontier = set(); path = []
                        nodes_visited = path_cost = 0; elapsed = 0.0
                        grid = build_grid()

                for i, r in enumerate(btn_rects['heur']):
                    if r.collidepoint(mx, my):
                        heur_idx = i
                        state = 'idle'; gen = None
                        visited = set(); frontier = set(); path = []
                        nodes_visited = path_cost = 0; elapsed = 0.0
                        grid = build_grid()

                if btn_rects['run'].collidepoint(mx, my) and state in ('idle','done','no_path'):
                    grid = build_grid()
                    visited = set(); frontier = set(); path = []
                    nodes_visited = path_cost = 0; elapsed = 0.0
                    h_fn = manhattan if heur_idx == 0 else euclidean
                    if algo_idx == 0:
                        gen = gbfs(grid, START_POS, GOAL_POS, h_fn)
                    else:
                        gen = astar(grid, START_POS, GOAL_POS, h_fn)
                    state = 'running'

                if btn_rects['reset'].collidepoint(mx, my):
                    grid = build_grid()
                    visited = set(); frontier = set(); path = []
                    nodes_visited = path_cost = 0; elapsed = 0.0
                    state = 'idle'; gen = None

                # Toggle walls by clicking grid cells
                if mx < COLS*CELL and state in ('idle',):
                    r = my // CELL
                    c = mx // CELL
                    if grid[r][c] == EMPTY:
                        grid[r][c] = WALL
                    elif grid[r][c] == WALL:
                        grid[r][c] = EMPTY

        # Advance search animation
        if state == 'running' and gen and (now - last_step) >= STEP_DELAY:
            last_step = now
            try:
                step = next(gen)
                kind, _, vis, fron, pth, nv, pc, el = step
                visited  = vis
                frontier = fron
                nodes_visited = nv
                if kind == 'done':
                    elapsed = el
                    if pth:
                        path = pth
                        path_cost = pc
                        state = 'done'
                    else:
                        state = 'no_path'
                    gen = None
            except StopIteration:
                state = 'done'; gen = None

        # ── Draw ──────────────────────────────────────────────────────────────
        surf.fill(BLACK)
        path_set = set(path) if path else set()
        draw_grid(surf, grid, visited, frontier, path_set)
        draw_panel(surf, font_big, font_med, font_sm,
                   algo_idx, heur_idx, nodes_visited, path_cost, elapsed,
                   btn_rects, hover_idx, algo_idx, heur_idx, state)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
