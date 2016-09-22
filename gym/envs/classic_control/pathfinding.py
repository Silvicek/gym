import heapq
import numpy as np

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

cost = dist

def goal_reached(a, b):
    return dist(a, b) < 20.

def a_star_search(start, goal, neighbors):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    final_state = (0., 0.)

    while not frontier.empty():
        current = frontier.get()

        if goal_reached(current, goal):
            final_state = current
            break

        for next in neighbors(current):
            new_cost = cost_so_far[current] + cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + dist(goal, next)
                frontier.put(next, priority)
                came_from[next] = current



    current = final_state
    trajectory = []
    if final_state == (0.,0.):
        return trajectory
    while True:
        if current == start:
            break
        trajectory.append(current)
        current = came_from[current]

    return trajectory


def plus(x, y):
    return x[0]+y[0], x[1]+y[1]

neighbors = [(0, 10), (10, 10), (10, 0), (-10, 0), (-10, -10), (0, -10)]
# neighbors = [(0, 1), (1, 1), (1, 0), (-1, 0), (-1, -1), (0, -1)]


class Trajectory:

    def __init__(self, start, target, check_availability):
        self.check = check_availability

        self.trajectory = a_star_search(tuple(start), tuple(target), self.available_nodes)
        # for i, point in enumerate(self.trajectory):
        #     self.trajectory[i] = (int(self.trajectory[i][0]), int(self.trajectory[i][1]))

    def available_nodes(self, position):
        return [plus(position, x) for x in neighbors if self.check(plus(position, x))]

    def distance(self, position):
        if len(self.trajectory) < 1:
            return 0.
        d = 1000.
        for x in self.trajectory:
            if dist(position, x) < d:
                d = dist(position, x)
        return d

