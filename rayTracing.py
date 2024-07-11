import numpy as np
from PIL import Image
from IPython.display import display

def normalize(v):
    return v / np.linalg.norm(v)

def reflect(I, N):
    return I - 2 * np.dot(I, N) * N

def intersect_sphere(O, D, S, R):
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        dist_sqrt = np.sqrt(disc)
        q = -0.5 * (b + np.sign(b) * dist_sqrt)
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return None

width, height = 800, 600
max_depth = 5
O = np.array([0.0, 0.0, 0.0])
Q = np.array([0.0, 0.0, 1.0])
light = np.array([5.0, 5.0, -10.0])
L = normalize(light - O)
color_light = np.array([1.0, 1.0, 1.0])
ambient = 0.05

spheres = [
    {"center": np.array([0.0, -1.0, 3.0]), "radius": 1.0, "color": np.array([1.0, 0.0, 0.0]), "reflection": 0.5},
    {"center": np.array([2.0, 0.0, 4.0]), "radius": 1.0, "color": np.array([0.0, 0.0, 1.0]), "reflection": 0.5},
    {"center": np.array([-2.0, 0.0, 4.0]), "radius": 1.0, "color": np.array([0.0, 1.0, 0.0]), "reflection": 0.5},
]

def trace_ray(O, D, depth):
    if depth >= max_depth:
        return np.zeros(3)

    t_min = np.inf
    sphere_hit = None

    for sphere in spheres:
        t = intersect_sphere(O, D, sphere["center"], sphere["radius"])
        if t and t < t_min:
            t_min = t
            sphere_hit = sphere

    if not sphere_hit:
        return np.zeros(3)

    P = O + D * t_min
    N = normalize(P - sphere_hit["center"])
    color = sphere_hit["color"] * ambient

    to_light = normalize(light - P)
    if np.dot(N, to_light) > 0:
        color += sphere_hit["color"] * color_light * max(np.dot(N, to_light), 0)

    if sphere_hit["reflection"] > 0:
        R = reflect(D, N)
        color += sphere_hit["reflection"] * trace_ray(P + N * 0.001, R, depth + 1)

    return color

img = Image.new("RGB", (width, height))
pixels = img.load()

for i in range(width):
    for j in range(height):
        x = (2 * (i + 0.5) / width - 1) * np.tan(np.pi / 6) * width / height
        y = -(2 * (j + 0.5) / height - 1) * np.tan(np.pi / 6)
        D = normalize(np.array([x, y, 1]) - O)
        color = trace_ray(O, D, 0)
        color = np.clip(color, 0, 1)
        pixels[i, j] = tuple((color * 255).astype(int))

display(img)
img.save("ray_tracing_reflection.png")