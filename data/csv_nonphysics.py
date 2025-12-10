from ursina import *
import pandas as pd

# Configuration of game space
GRID_SIZE = 10  # 10x10x10 cube
CUBE_SCALE = 1
SNAKE_LENGTH = 3
MOVE_INTERVAL = 0.1  # seconds between frames
CSV_FILE = "episode_actions.csv"  # <-- your CSV file here

# loads the csv
df = pd.read_csv(CSV_FILE)

# global variables
snake_positions = []
current_row_idx = 0
time_acc = 0

# starting ursina
app = Ursina()
window.title = "3D Snake CSV Playback"
window.borderless = False
window.fullscreen = False
window.exit_button.visible = True
window.fps_counter.enabled = True

# camera start and configuration
camera.position = (GRID_SIZE / 2, GRID_SIZE / 2, -GRID_SIZE * 1.5)
camera.look_at(Vec3(GRID_SIZE / 2, GRID_SIZE / 2, GRID_SIZE / 2))
camera.fov = 60

# text to display live episode number
episode_text = Text(
    text="Episode: 0",
    position=(-0.48, 0.45),   # x, y
    origin=(0, 0),
    scale=2
)


# creating a cube to visualize the game space.
def create_wireframe_cube(size):
    """Create a wireframe cube mesh"""
    half = size / 2

    # 8 corners of the cube
    corners = [
        Vec3(-half, -half, -half),  # 0: bottom-left-front
        Vec3(half, -half, -half),  # 1: bottom-right-front
        Vec3(half, half, -half),  # 2: top-right-front
        Vec3(-half, half, -half),  # 3: top-left-front
        Vec3(-half, -half, half),  # 4: bottom-left-back
        Vec3(half, -half, half),  # 5: bottom-right-back
        Vec3(half, half, half),  # 6: top-right-back
        Vec3(-half, half, half),  # 7: top-left-back
    ]

    # the 12 edges of the cube
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    vertices = []
    for start, end in edges:
        vertices.append(corners[start])
        vertices.append(corners[end])

    return Mesh(vertices=vertices, mode='line')


# starting and configuring the game space
wireframe = Entity(
    model=create_wireframe_cube(GRID_SIZE),
    color=color.white,
    position=(GRID_SIZE / 2, GRID_SIZE / 2, GRID_SIZE / 2)
)


board = Entity(
    model='cube',
    scale=GRID_SIZE,
    color=color.rgba(50, 50, 50, 30),
    origin=(0.5, 0.5, 0.5)
)

# apple
apple = Entity(model='cube', color=color.red, scale=CUBE_SCALE)

# snake body (list of cubes)
snake = [Entity(model='cube', color=color.green, scale=CUBE_SCALE) for _ in range(SNAKE_LENGTH)]


# helper functions
def player_has_next():
    global current_row_idx
    return current_row_idx < len(df)


def player_step():
    global current_row_idx
    if current_row_idx >= len(df):
        return None
    row = df.iloc[current_row_idx]
    current_row_idx += 1
    return row


def reset_episode():
    global snake_positions
    snake_positions.clear()
    for seg in snake:
        seg.enabled = False
    apple.position = (0, 0, 0)


# function for updates
def update():
    global time_acc, snake_positions, apple, current_row_idx

    if not player_has_next():
        return

    time_acc += time.dt

    while time_acc >= MOVE_INTERVAL:
        time_acc -= MOVE_INTERVAL

        row = player_step()
        if row is None:
            return

        # skipping non uniform rows
        if pd.isna(row.head_x) or pd.isna(row.head_y) or pd.isna(row.head_z):
            continue


        try:
            ep = int(row.episode)
        except:
            ep = 0
        episode_text.text = f"Episode: {ep}"

        # head positioning
        hx = int(row.head_x)
        hy = int(row.head_y)
        hz = int(row.head_z)
        head_pos = (hx, hy, hz)

        # append to snake history
        snake_positions.append(head_pos)

        # trims snake body to fixed length
        while len(snake_positions) > SNAKE_LENGTH:
            snake_positions.pop(0)


        for i, segment in enumerate(snake):
            if i < len(snake_positions):
                segment.position = snake_positions[i]
                segment.enabled = True
            else:
                segment.enabled = False


        try:
            fx = int(row.food_x)
            fy = int(row.food_y)
            fz = int(row.food_z)
            apple.position = (fx, fy, fz)
        except:
            pass


        if bool(row.done):
            reset_episode()


# runs the game
app.run()