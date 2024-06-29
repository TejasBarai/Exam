import numpy as np

class LineModel:
    def __init__(self):
        self.m = 0  # Default slope
        self.c = 0  # Default intercept
        self.points = []

    def add_points(self, points):
       self.points.extend(points)

    def total_distance(self):
        return sum(self.calculate_distance(point) for point in self.points)

    def calculate_distance(self, point):
        x, y = point
        return abs(self.m * x - y + self.c) / np.sqrt(self.m**2 + 1)

    def line_equation(self):
        return f"y = {self.m}x + {self.c}"


class OptimizeLineModel(LineModel):

    def __init__(self, learning_rate, iterations):
        super().__init__()
        self.learning_rate = learning_rate
        self.iterations = iterations

    def optimize_line(self):
        for _ in range(self.iterations):
            dm = 0  # Gradient for slope
            dc = 0  # Gradient for intercept
            for point in self.points:
                x, y = point
                distance = self.calculate_distance(point)
                sign = np.sign(self.m * x - y + self.c)
                dm += sign * x / np.sqrt(self.m**2 + 1)
                dc += sign / np.sqrt(self.m**2 + 1)
            
            self.m -= self.learning_rate * dm
            self.c -= self.learning_rate * dc


# TestCase
points = [(1, 2), (4, 8), (3, 6), (8, 16)]

model = OptimizeLineModel(learning_rate=0.5, iterations=10)
model.add_points(points)
model.optimize_line()

print(f'Total distance: {model.total_distance()}')
print(f'Line equation is: {model.line_equation()}')
