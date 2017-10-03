import numpy as np

"""Graph Model: this program is to get the graph model from skeleton images"""


class Skeleton2Graph(object):
    def __init__(self, Img):
        self.Img = Img
        self.weights = {}

    def get_all_nodes(self):
        all_nodes = []
        for x in range(self.Img.shape[0]):
            for y in range(self.Img.shape[1]):
                if self.Img[x, y] == 1:
                    all_nodes.append((x,y))
        return all_nodes


    def neighbors(self, node):
        (x, y) = node
        all_points = self.get_all_nodes()
        dirs = [[-1,-1], [-1, 1], [1, -1], [1,1], [-1,0],  [0, -1], [0, 1],  [1,0]]
        result = []
        # self.weights[node] = []
        for index, dir in enumerate(dirs):
            neighbor = (node[0]+dir[0], node[1]+dir[1])
            if neighbor in all_points:
                result.append(neighbor)
                if index <= 3:
                    self.weights[node, neighbor] = 1.414
                else:
                    self.weights[node, neighbor] = 1

        return result

    def cost(self, from_node, to_node):
        return self.weights[from_node, to_node]
        # return self.weights.get(to_node, 1)


#######dijkstra_search##################################
import heapq
class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) ==0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def dijkstra_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current

    if goal in came_from:
        flag = True
        cost = cost_so_far[goal]
        path = [goal, came_from[goal]]
        while start not in path:
            path.append(came_from[path[-1]])
        path = path[::-1]


    else:
        flag = False
        cost = -1
        path = [(-1, -1)]
        # path = np.array([[-1], [-1]])

    return flag, cost, path


#####Breadth first Search####################

import collections

class Queue:
    def __init__(self):
        self.elements = collections.deque()

    def empty(self):
        return len(self.elements) == 0

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.popleft()


def breadth_first_search(graph, start, goal):
    frontier = Queue()
    frontier.put(start)
    came_from = {}
    came_from[start] = None

    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            break

        for next in graph.neighbors(current):
            if next not in came_from:
                frontier.put(next)
                came_from[next] = current

    connected_component = came_from.keys()
    return connected_component


def breadth_first_search2(graph, start):
    frontier = Queue()
    frontier.put(start)
    came_from = {}
    came_from[start] = None

    while not frontier.empty():
        current = frontier.get()
        for next in graph.neighbors(current):
            if next not in came_from:
                frontier.put(next)
                came_from[next] = current

    connected_component = came_from.keys()
    return connected_component


def depth_first_search2(graph, start):
    visited, stack = [], [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.append(vertex)
            stack.extend(set(graph.neighbors(vertex)) - set(visited))
    return visited