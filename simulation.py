from taichi.types import compound_types
from taichi.types import compound_types
import taichi as ti

ti.init(arch=ti.cuda)

WIDTH = 2000
HEIGHT = 1000

MAX_PREY = 9000
MAX_PRED = 1000

RESTITUTION = 1.0

PREY_SPEED = 5.0
PRED_SPEED = 5.0

CELL_SIZE = 25
SEPARATION = CELL_SIZE * 0.75
GRID_W = WIDTH//CELL_SIZE
GRID_H = HEIGHT//CELL_SIZE
MAX_PER_CELL = 10

MAX_FLOCK_MEMBERS = 10

num_active_prey = ti.field(dtype=ti.i32, shape=())
free_top_prey = ti.field(dtype=ti.i32, shape=())

num_active_pred = ti.field(dtype=ti.i32, shape=())
free_top_pred = ti.field(dtype=ti.i32, shape=())

prey_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PREY)
prey_vel = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PREY)

pred_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PRED)
pred_vel = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PRED)

prey_health = ti.field(dtype=ti.f32, shape=(MAX_PREY))
prey_max_health = ti.field(dtype=ti.f32, shape=(MAX_PREY))
prey_panic = ti.field(dtype=ti.f32, shape=(MAX_PREY))
prey_is_alive = ti.field(dtype=ti.i32, shape=(MAX_PREY))
prey_free_stack = ti.field(dtype=ti.i32, shape=(MAX_PREY))
prey_active_indices = ti.field(dtype=ti.i32, shape=(MAX_PREY))

pred_health = ti.field(dtype=ti.f32, shape=(MAX_PRED))
pred_max_health = ti.field(dtype=ti.f32, shape=(MAX_PRED))
pred_hunger = ti.field(dtype=ti.i32, shape=(MAX_PRED))
pred_target_id = ti.field(dtype=ti.i32, shape=(MAX_PRED))
pred_is_alive = ti.field(dtype=ti.i32, shape=(MAX_PRED))
pred_free_stack = ti.field(dtype=ti.i32, shape=(MAX_PRED))
pred_active_indices = ti.field(dtype=ti.i32, shape=(MAX_PRED))

render_prey_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PREY)
render_pred_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PRED)

grid_count = ti.field(ti.i32, shape=(GRID_W, GRID_H))
grid_cells = ti.field(ti.i32, shape=(GRID_W, GRID_H, MAX_PER_CELL))

@ti.kernel
def init_simulation():
    num_active_prey[None] = 0
    num_active_pred[None] = 0
    
    for i in range(MAX_PREY):
        prey_free_stack[i] = i
        prey_is_alive[i] = 0
    free_top_prey[None] = MAX_PREY
    
    for i in range(MAX_PRED):
        pred_free_stack[i] = i
        pred_is_alive[i] = 0
    free_top_pred[None] = MAX_PRED

@ti.kernel
def _spawn_prey_kernel(count: ti.i32):
    for i in range(count):
        slot = prey_free_stack[ti.atomic_sub(free_top_prey[None], 1) - 1]
        prey_active_indices[ti.atomic_add(num_active_prey[None], 1)] = slot
        prey_health[slot] = 1000
        prey_max_health[slot] = 1000
        prey_pos[slot] = ti.Vector([WIDTH * ti.random(), HEIGHT * ti.random()])
        prey_vel[slot] = ti.Vector([(ti.random() - 0.5) * 2.0 * PREY_SPEED, (ti.random() - 0.5) * 2.0 * PREY_SPEED])
        prey_is_alive[slot] = True

@ti.kernel
def _spawn_pred_kernel(count: ti.i32):
    for i in range(count):
        slot = pred_free_stack[ti.atomic_sub(free_top_pred[None], 1) - 1]
        pred_active_indices[ti.atomic_add(num_active_pred[None], 1)] = slot
        pred_health[slot] = 1000
        pred_max_health[slot] = 1000
        pred_pos[slot] = ti.Vector([WIDTH * ti.random(), HEIGHT * ti.random()])
        pred_vel[slot] = ti.Vector([(ti.random() - 0.5) * 2.0 * PRED_SPEED, (ti.random() - 0.5) * 2.0 * PRED_SPEED])
        pred_is_alive[slot] = True

@ti.kernel
def update_prey_motion(dt: ti.f32):
    for i in range(0, num_active_prey[None]):
        slot = prey_active_indices[i]

        if ti.random() < 0.05:
            perp = ti.Vector([-prey_vel[slot].y, prey_vel[slot].x]).norm()
            side = 1.0 if ti.random() < 0.5 else -1.0
            prey_vel[slot] += perp * side * PREY_SPEED * 0.8

        cx = ti.cast(prey_pos[slot].x / CELL_SIZE, ti.i32)
        cy = ti.cast(prey_pos[slot].y / CELL_SIZE, ti.i32)

        steer_separation = ti.Vector([0.0, 0.0])
        sum_vel = ti.Vector([0.0, 0.0])
        sum_pos = ti.Vector([0.0, 0.0])
        neighbor_count = 0
        total_force = ti.Vector([0.0, 0.0])

        for dx in range(-1, 2):
            for dy in range(-1,2):
                nx = cx + dx
                ny = cy + dy
                if nx >= 0 and nx < GRID_W and ny >= 0 and ny < GRID_H:
                    for k in range(min(grid_count[nx, ny], MAX_PER_CELL)):
                        neighbor_slot = grid_cells[nx, ny, k]
                        if neighbor_slot != slot:
                            diff = prey_pos[slot] - prey_pos[neighbor_slot]
                            dist = diff.norm()
                            if dist < CELL_SIZE:
                                neighbor_count += 1
                                sum_pos += prey_pos[neighbor_slot]
                                sum_vel += prey_vel[neighbor_slot]
                                if dist < SEPARATION:
                                    steer_separation += diff / dist * (1 - dist / SEPARATION)
        
        if neighbor_count > 0:
            avg_vel = sum_vel / neighbor_count
            steer_alignment = avg_vel - prey_vel[slot]
            center = sum_pos / neighbor_count
            steer_cohesion = center - prey_pos[slot]
            if neighbor_count > MAX_FLOCK_MEMBERS:
                steer_cohesion *= -0.1
            total_force = (4.0 * steer_separation) + (0.8 * steer_alignment) + (0.5 * steer_cohesion)

        prey_vel[slot] += total_force

        next_x = prey_pos[slot].x + prey_vel[slot].x + (prey_vel[slot].x * dt)
        if next_x < 0.0 or next_x > WIDTH:
            prey_vel[slot].x *= -RESTITUTION
        
        next_y = prey_pos[slot].y + prey_vel[slot].y + (prey_vel[slot].y * dt)
        if next_y < 0.0 or next_y > HEIGHT:
            prey_vel[slot].y *= -RESTITUTION
        
        prey_vel[slot] += prey_vel[slot] * dt

        prey_vel[slot] = (prey_vel[slot] / prey_vel[slot].norm()) * PREY_SPEED

        prey_pos[slot] += prey_vel[slot]
        prey_pos[slot].x = ti.math.clamp(prey_pos[slot].x, 0.0, WIDTH)
        prey_pos[slot].y = ti.math.clamp(prey_pos[slot].y, 0.0, HEIGHT)

@ti.kernel
def update_pred_motion(dt: ti.f32):
    for i in range(0, num_active_pred[None]):
        slot = pred_active_indices[i]
        alpha = pred_active_indices[0]

        drift = ti.Vector([(ti.random() - 0.5) * 2.0, (ti.random() - 0.5) * 2.0])

        next_x = pred_pos[slot].x + pred_vel[slot].x + (pred_vel[slot].x * dt) + drift[0]
        if next_x < 0.0 or next_x > WIDTH:
            pred_vel[slot].x *= -RESTITUTION
        
        next_y = pred_pos[slot].y + pred_vel[slot].y + (pred_vel[slot].y * dt) + drift[1]
        if next_y < 0.0 or next_y > HEIGHT:
            pred_vel[slot].y *= -RESTITUTION
        
        # prey_vel[slot] += prey_pos[slot] * dt

        pred_pos[slot] += pred_vel[slot] + (pred_vel[slot] * dt) + drift
        pred_pos[slot].x = ti.math.clamp(pred_pos[slot].x, 0.0, WIDTH)
        pred_pos[slot].y = ti.math.clamp(pred_pos[slot].y, 0.0, HEIGHT)


def spawn_prey(count):
    if free_top_prey[None] >= count:
        _spawn_prey_kernel(count)
    else:
        raise RuntimeError("free slot not enough for spawning")

def spawn_pred(count):
    if free_top_pred[None] >= count:
        _spawn_pred_kernel(count)
    else:
        raise RuntimeError("free slot not enough for spawning")

@ti.kernel
def normalize_pos_prey():
    for i in range(num_active_prey[None]):
        slot = prey_active_indices[i]
        x_norm = prey_pos[slot].x / WIDTH
        y_norm = prey_pos[slot].y / HEIGHT
        render_prey_pos[i] = ti.Vector([x_norm, y_norm])

@ti.kernel
def normalize_pos_pred():
    for i in range(num_active_pred[None]):
        slot = pred_active_indices[i]
        x_norm = pred_pos[slot].x / WIDTH
        y_norm = pred_pos[slot].y / HEIGHT
        render_pred_pos[i] = ti.Vector([x_norm, y_norm])

@ti.kernel
def clear_grid():
    for cx in range(GRID_W):
        for cy in range(GRID_H):
            grid_count[cx, cy] = 0

@ti.kernel
def bin_kernel():
    for i in range(num_active_prey[None]):
        slot = prey_active_indices[i]
        cx = ti.cast(prey_pos[slot].x / CELL_SIZE, ti.i32)
        cy = ti.cast(prey_pos[slot].y / CELL_SIZE, ti.i32)
        cx = ti.math.clamp(cx, 0, GRID_W-1)
        cy = ti.math.clamp(cy, 0, GRID_H-1)
        idx = ti.atomic_add(grid_count[cx, cy], 1)
        
        if idx < MAX_PER_CELL:
            grid_cells[cx, cy, idx] = slot

def main():
    init_simulation()
    spawn_prey(1000)
    spawn_pred(10)
    
    gui = ti.ui.Window("Life Simulation", res=(WIDTH, HEIGHT))
    
    dt = 1.0/60.0

    while gui.running:
        clear_grid()
        bin_kernel()
        update_prey_motion(dt)
        update_pred_motion(dt)
        normalize_pos_prey()
        normalize_pos_pred()
        active_count = num_active_prey[None]
        canvas = gui.get_canvas()
        canvas.circles(render_prey_pos, radius=0.005, color=(0.0, 1.0, 0.0))
        canvas.circles(render_pred_pos, radius=0.005, color=(1.0, 0.0, 0.0))

        gui.show()


main()