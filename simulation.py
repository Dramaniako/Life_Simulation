from taichi.types import compound_types
from taichi.types import compound_types
import taichi as ti

ti.init(arch=ti.cuda)

WIDTH = 1280
HEIGHT = 720

MAX_PREY = 9000
MAX_PRED = 1000

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


def spawn_prey(count):
    if free_top_prey[None] >= count:
        _spawn_prey_kernel(count)
    else:
        raise RuntimeError("free slot not enough for spawning")


init_simulation()
spawn_prey(50)
print(num_active_prey[None])