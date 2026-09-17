from taichi.types import compound_types
from taichi.types import compound_types
import taichi as ti

ti.init(arch=ti.cuda)

WIDTH = 1280
HEIGHT = 720

MAX_PREY = 9000
MAX_PRED = 1000

RESTITUTION = 1.0

PREY_SPEED = 150.0

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

pred_hunger = ti.field(dtype=ti.i32, shape=(MAX_PRED))
pred_target_id = ti.field(dtype=ti.i32, shape=(MAX_PRED))
pred_is_alive = ti.field(dtype=ti.i32, shape=(MAX_PRED))
pred_free_stack = ti.field(dtype=ti.i32, shape=(MAX_PRED))
pred_active_indices = ti.field(dtype=ti.i32, shape=(MAX_PRED))

render_prey_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PREY)

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
        prey_vel[slot] = ti.Vector([(ti.random() - 0.5) * 2.0, (ti.random() - 0.5) * 2.0])
        prey_is_alive[slot] = True

@ti.kernel
def update_prey_motion(dt: ti.f32):
    for i in range(1, num_active_prey[None]):
        slot = prey_active_indices[i]
        alpha = prey_active_indices[0]

        next_x = prey_pos[slot].x + prey_vel[slot].x + prey_pos[alpha].x * dt * 100
        if next_x < 0.0 or next_x > WIDTH or prey_vel[slot].x == WIDTH:
            prey_vel[slot].x *= -RESTITUTION
        
        next_y = prey_pos[slot].y + prey_vel[slot].y + prey_pos[alpha].y * dt * 100
        if next_y < 0.0 or next_y > HEIGHT or prey_vel[slot].y == HEIGHT:
            prey_vel[slot].y *= -RESTITUTION
        

        prey_pos[slot] += prey_vel[slot] * dt + prey_pos[alpha] * dt
        prey_pos[slot].x = ti.math.clamp(prey_pos[slot].x, 0.0, WIDTH)
        prey_pos[slot].y = ti.math.clamp(prey_pos[slot].y, 0.0, HEIGHT)


def spawn_prey(count):
    if free_top_prey[None] >= count:
        _spawn_prey_kernel(count)
    else:
        raise RuntimeError("free slot not enough for spawning")

@ti.kernel
def normalize_pos():
    for i in range(num_active_prey[None]):
        slot = prey_active_indices[i]
        x_norm = prey_pos[slot].x / WIDTH
        y_norm = prey_pos[slot].y / HEIGHT
        render_prey_pos[i] = ti.Vector([x_norm, y_norm])

def main():
    init_simulation()
    spawn_prey(5000)
    
    gui = ti.ui.Window("Life Simulation", res=(WIDTH, HEIGHT))
    
    dt = 1.0/60.0

    while gui.running:
        update_prey_motion(dt)
        normalize_pos()
        active_count = num_active_prey[None]
        canvas = gui.get_canvas()
        canvas.circles(render_prey_pos, radius=0.005, color=(0.0, 1.0, 0.0))
        gui.show()


main()