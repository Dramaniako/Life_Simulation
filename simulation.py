from taichi.math import clamp
import taichi as ti

ti.init(arch=ti.cuda)

vec2 = ti.types.vector(2, ti.f32)

WIDTH = 2000
HEIGHT = 1000

MAX_PREY = 9000
MAX_PRED = 1000

RESTITUTION = 1.0

PREY_SPEED = 2.5
PRED_SPEED = 5.0

CELL_SIZE = 25
SEPARATION = CELL_SIZE * 0.75
GRID_W = WIDTH//CELL_SIZE
GRID_H = HEIGHT//CELL_SIZE
MAX_PER_CELL = 10

MAX_FLOCK_MEMBERS = 10

PREY_REPRODUCTION_ENERGY = 1200
PRED_REPRODUCTION_ENERGY = 3

PRED_VISION_RADIUS = 200.0
EAT_RADIUS = 8.0
PREY_FLEE_RADIUS = 120.0

SEARCH_CELLS = int(PRED_VISION_RADIUS // CELL_SIZE) + 1

PRED_MAX_HUNGER = 2400
PRED_HUNGER_RATE = 1
PRED_ACCEL = 2.0
HUNGER_HUNT_THRESHOLD = 400
PRED_EAT_ENERGY_GAIN = 2400

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
prey_energy = ti.field(dtype=ti.i32, shape=(MAX_PREY))

pred_health = ti.field(dtype=ti.f32, shape=(MAX_PRED))
pred_max_health = ti.field(dtype=ti.f32, shape=(MAX_PRED))
pred_hunger = ti.field(dtype=ti.i32, shape=(MAX_PRED))
pred_target_id = ti.field(dtype=ti.i32, shape=(MAX_PRED))
pred_is_alive = ti.field(dtype=ti.i32, shape=(MAX_PRED))
pred_free_stack = ti.field(dtype=ti.i32, shape=(MAX_PRED))
pred_active_indices = ti.field(dtype=ti.i32, shape=(MAX_PRED))
pred_energy = ti.field(dtype=ti.i32, shape=(MAX_PRED))

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
        prey_energy[slot] = 0
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
        pred_energy[slot] = 0
        pred_hunger[slot] = 0

@ti.kernel
def update_prey_motion(dt: ti.f32):
    for i in range(0, num_active_prey[None]):
        slot = prey_active_indices[i]

        cx = ti.cast(prey_pos[slot].x / CELL_SIZE, ti.i32)
        cy = ti.cast(prey_pos[slot].y / CELL_SIZE, ti.i32)

        steer_separation = ti.Vector([0.0, 0.0])
        sum_vel = ti.Vector([0.0, 0.0])
        sum_pos = ti.Vector([0.0, 0.0])
        neighbor_count = 0
        total_force = ti.Vector([0.0, 0.0])

        for dx in range(-1, 2):
            for dy in range(-1, 2):
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
                steer_cohesion = ti.Vector([0.0, 0.0])
            total_force = (4.0 * steer_separation) + (0.8 * steer_alignment) + (0.5 * steer_cohesion)

        flee_force = ti.Vector([0.0, 0.0])
        is_panicking = 0

        for p_idx in range(num_active_pred[None]):
            p_slot = pred_active_indices[p_idx]
            diff = prey_pos[slot] - pred_pos[p_slot]
            dist = diff.norm()
            if dist < PREY_FLEE_RADIUS and dist > 0.001:
                flee_force += (diff / dist) * (1.0 - dist / PREY_FLEE_RADIUS)
                is_panicking = 1

        total_force += flee_force * 10.0

        margin = 60.0
        turn_strength = 2.0
        if prey_pos[slot].x < margin:
            total_force.x += turn_strength * (1.0 - prey_pos[slot].x / margin)
        elif prey_pos[slot].x > WIDTH - margin:
            total_force.x -= turn_strength * (1.0 - (WIDTH - prey_pos[slot].x) / margin)
        if prey_pos[slot].y < margin:
            total_force.y += turn_strength * (1.0 - prey_pos[slot].y / margin)
        elif prey_pos[slot].y > HEIGHT - margin:
            total_force.y -= turn_strength * (1.0 - (HEIGHT - prey_pos[slot].y) / margin)

        prey_vel[slot] += total_force

        target_speed = PREY_SPEED * 1.4 if is_panicking else PREY_SPEED

        vel_norm = prey_vel[slot].norm()
        if vel_norm > 0.001:
            prey_vel[slot] = (prey_vel[slot] / vel_norm) * target_speed

        prey_pos[slot] += prey_vel[slot]

        prey_pos[slot].x = ti.math.clamp(prey_pos[slot].x, 0.0, WIDTH)
        prey_pos[slot].y = ti.math.clamp(prey_pos[slot].y, 0.0, HEIGHT)
        if is_panicking == 0 and neighbor_count < 10:
            prey_energy[slot] += 1


@ti.kernel
def update_pred_motion(dt: ti.f32):
    for i in range(num_active_pred[None]):
        slot = pred_active_indices[i]

        pred_hunger[slot] = ti.min(pred_hunger[slot] + PRED_HUNGER_RATE, PRED_MAX_HUNGER)

        is_hunting = pred_hunger[slot] >= HUNGER_HUNT_THRESHOLD

        closest_prey = -1
        closest_dist = PRED_VISION_RADIUS

        if is_hunting:
            cx = ti.cast(pred_pos[slot].x / CELL_SIZE, ti.i32)
            cy = ti.cast(pred_pos[slot].y / CELL_SIZE, ti.i32)
            
            for dx in range(-SEARCH_CELLS, SEARCH_CELLS + 1):
                for dy in range(-SEARCH_CELLS, SEARCH_CELLS + 1):
                    nx = cx + dx
                    ny = cy + dy
                    if nx >= 0 and nx < GRID_W and ny >= 0 and ny < GRID_H:
                        for k in range(min(grid_count[nx, ny], MAX_PER_CELL)):
                            prey_slot = grid_cells[nx, ny, k]
                            if prey_is_alive[prey_slot] == 1:
                                diff = prey_pos[prey_slot] - pred_pos[slot]
                                dist = diff.norm()
                                if dist < closest_dist:
                                    closest_dist = dist
                                    closest_prey = prey_slot
            
        if closest_prey != -1:
            diff = prey_pos[closest_prey] - pred_pos[slot]
            if closest_dist <= EAT_RADIUS:
                if ti.atomic_and(prey_is_alive[closest_prey], 0) == 1:
                    pred_hunger[slot] = max(0, pred_hunger[slot] - PRED_EAT_ENERGY_GAIN)
                    pred_health[slot] = pred_max_health[slot]
                    pred_target_id[slot] = -1
                    pred_energy[slot] += 1
            else:
                dir = diff / closest_dist
                pred_vel[slot] += dir * PRED_ACCEL
        else:
            center = ti.Vector([WIDTH * 0.5, HEIGHT * 0.5])
            to_center = center - pred_pos[slot]
            dist_to_center = to_center.norm()
            if dist_to_center > 60.0:
                dir_to_center = to_center / dist_to_center
                pred_vel[slot] += dir_to_center * 0.5
            else:
                wander = ti.Vector([(ti.random() - 0.5) * 2.0, (ti.random() - 0.5) * 2.0])
                pred_vel[slot] += wander * 0.2

        max_speed = PRED_SPEED if is_hunting else (PRED_SPEED * 0.5)
        speed = pred_vel[slot].norm()
        if speed > max_speed:
            pred_vel[slot] = (pred_vel[slot] / speed) * max_speed
        elif speed < 0.1:
            pred_vel[slot] = ti.Vector([1.0, 0.0]) * max_speed

        pred_pos[slot] += pred_vel[slot]

        if pred_pos[slot].x < 0.0 or pred_pos[slot].x > WIDTH:
            pred_vel[slot].x += -RESTITUTION
        if pred_pos[slot].y < 0.0 or pred_pos[slot].y > HEIGHT:
            pred_vel[slot].y += -RESTITUTION

        pred_pos[slot].x = ti.math.clamp(pred_pos[slot].x, 0.0, WIDTH)
        pred_pos[slot].y = ti.math.clamp(pred_pos[slot].y, 0.0, HEIGHT)

        if pred_hunger[slot] >= PRED_MAX_HUNGER:
            pred_is_alive[slot] = 0

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
    for i in range(MAX_PREY):
        if i < num_active_prey[None]:
            slot = prey_active_indices[i]
            x_norm = prey_pos[slot].x / WIDTH
            y_norm = prey_pos[slot].y / HEIGHT
            render_prey_pos[i] = ti.Vector([x_norm, y_norm])
        else:
            render_prey_pos[i] = ti.Vector([-10.0, -10.0])

@ti.kernel
def normalize_pos_pred():
    for i in range(MAX_PRED):
        if i < num_active_pred[None]:
            slot = pred_active_indices[i]
            x_norm = pred_pos[slot].x / WIDTH
            y_norm = pred_pos[slot].y / HEIGHT
            render_pred_pos[i] = ti.Vector([x_norm, y_norm])
        else:
            render_pred_pos[i] = ti.Vector([-10.0, -10.0])

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

@ti.kernel
def cleanup_dead_prey():
    for _ in range(1):
        surviving = 0
        for i in range(num_active_prey[None]):
            slot = prey_active_indices[i]
            if prey_is_alive[slot] == 1:
                prey_active_indices[surviving] = slot
                surviving += 1
            else:
                stack_top = free_top_prey[None]
                if stack_top >= 0 and stack_top < MAX_PREY:
                    prey_free_stack[stack_top] = slot
                    free_top_prey[None] += 1
        num_active_prey[None] = surviving

@ti.kernel
def cleanup_dead_pred():
    for _ in range(1):
        surviving = 0
        for i in range(num_active_pred[None]):
            slot = pred_active_indices[i]
            if pred_is_alive[slot] == 1:
                pred_active_indices[surviving] = slot
                surviving += 1
            else:
                stack_top = free_top_pred[None]
                if stack_top >= 0 and stack_top < MAX_PRED:
                    pred_free_stack[stack_top] = slot
                    free_top_pred[None] += 1
        num_active_pred[None] = surviving

@ti.func
def spawn_one_prey(parent_pos: vec2) -> ti.i32:
    success = 0
    if free_top_prey[None] > 0 and num_active_prey[None] < MAX_PREY:
        old_top = ti.atomic_sub(free_top_prey[None], 1)
        if old_top > 0:
            active_idx = ti.atomic_add(num_active_prey[None], 1)
            if active_idx < MAX_PREY:
                slot = prey_free_stack[old_top - 1]
                prey_active_indices[active_idx] = slot
                prey_health[slot] = 1000.0
                prey_max_health[slot] = 1000.0
                offset = ti.Vector([(ti.random() - 0.5) * 20.0, (ti.random() - 0.5) * 20.0])
                prey_pos[slot] = ti.math.clamp(parent_pos + offset, ti.Vector([0.0, 0.0]), ti.Vector([WIDTH, HEIGHT]))
                prey_vel[slot] = ti.Vector([(ti.random() - 0.5) * 2.0 * PREY_SPEED, (ti.random() - 0.5) * 2.0 * PREY_SPEED])
                prey_energy[slot] = 0
                prey_is_alive[slot] = 1
                success = 1
            else:
                ti.atomic_sub(num_active_prey[None], 1)
                ti.atomic_add(free_top_prey[None], 1)
        else:
            ti.atomic_add(free_top_prey[None], 1)
    return success

@ti.func
def spawn_one_pred(parent_pos: vec2) -> ti.i32:
    success = 0
    if free_top_pred[None] > 0 and num_active_pred[None] < MAX_PRED:
        old_top = ti.atomic_sub(free_top_pred[None], 1)
        if old_top > 0:
            active_idx = ti.atomic_add(num_active_pred[None], 1)
            if active_idx < MAX_PRED:
                slot = pred_free_stack[old_top - 1]
                pred_active_indices[active_idx] = slot
                pred_health[slot] = 1000.0
                pred_max_health[slot] = 1000.0
                offset = ti.Vector([(ti.random() - 0.5) * 20.0, (ti.random() - 0.5) * 20.0])
                pred_pos[slot] = ti.math.clamp(parent_pos + offset, ti.Vector([0.0, 0.0]), ti.Vector([WIDTH, HEIGHT]))
                pred_vel[slot] = ti.Vector([(ti.random() - 0.5) * 2.0 * PRED_SPEED, (ti.random() - 0.5) * 2.0 * PRED_SPEED])
                pred_energy[slot] = 0
                pred_hunger[slot] = 0
                pred_is_alive[slot] = 1
                success = 1
            else:
                ti.atomic_sub(num_active_pred[None], 1)
                ti.atomic_add(free_top_pred[None], 1)
        else:
            ti.atomic_add(free_top_pred[None], 1)
    return success

@ti.kernel
def prey_mating():
    for i in range(num_active_prey[None]):
        slot = prey_active_indices[i]
        if prey_energy[slot] >= PREY_REPRODUCTION_ENERGY:
            if spawn_one_prey(prey_pos[slot]) == 1:
                prey_energy[slot] = 0

@ti.kernel
def pred_mating():
    for i in range(num_active_pred[None]):
        slot = pred_active_indices[i]
        if pred_energy[slot] >= PRED_REPRODUCTION_ENERGY:
            if spawn_one_pred(pred_pos[slot]) == 1:
                pred_energy[slot] = 0
                                
def main():
    init_simulation()
    spawn_prey(300)
    spawn_pred(15)
    
    gui = ti.ui.Window("Life Simulation", res=(WIDTH, HEIGHT))
    
    dt = 1.0/240.0

    while gui.running:
        clear_grid()
        bin_kernel()
        update_pred_motion(dt)
        cleanup_dead_prey()
        cleanup_dead_pred()
        update_prey_motion(dt)
        prey_mating()
        pred_mating()
        normalize_pos_prey()
        normalize_pos_pred()
        active_count = num_active_prey[None]
        canvas = gui.get_canvas()
        canvas.circles(render_prey_pos, radius=0.002, color=(0.0, 1.0, 0.0))
        canvas.circles(render_pred_pos, radius=0.005, color=(1.0, 0.0, 0.0))

        window_gui = gui.get_gui()
        with window_gui.sub_window("Controls", x=0.02, y=0.02, width=0.18, height=0.22):
            window_gui.text(f"Prey: {num_active_prey[None]} / {MAX_PREY}")
            window_gui.text(f"Free: {free_top_prey[None]}")
            
            if window_gui.button("Spawn 1000 Prey"):
                to_spawn = min(1000, free_top_prey[None])
                if to_spawn > 0:
                    spawn_prey(to_spawn)
            
            window_gui.text(f"Pred: {num_active_pred[None]} / {MAX_PRED}")
            window_gui.text(f"Free: {free_top_pred[None]}")

            if window_gui.button("Spawn 10 Pred"):
                to_spawn = min(10, free_top_pred[None])
                if to_spawn > 0:
                    spawn_pred(to_spawn)

        gui.show()


if __name__ == "__main__":
    main()