import random
import math
import itertools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc


def dot_product(vec1, vec2):
    value = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    return value


def norm(vector):
    value = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    return value


def subtract(vec1, vec2):
    return [float(vec1[0]) - float(vec2[0]), float(vec1[1]) - float(vec2[1])]
    # return [float(vec1[0]),float(vec2[0])]


def scalmul(a, vec2):
    return [a * vec2[0], a * vec2[1]]


def throw():
    r = random.uniform(-1, 1)
    phi = random.uniform(-1, 1)  # * 2 * (math.pi)
    vector = [r, phi]
    return vector


def distance(vectors):
    for i, j in itertools.combinations(vectors, 2):
        displacement = [i[0] * math.cos(i[1]) - j[0] * math.cos(j[1]), i[0] * math.sin(i[1]) - j[0] * math.sin(j[1])]
        modulus = math.sqrt(float(displacement[0] ** 2 + displacement[1] ** 2))
        if modulus < 1:
            return False
    return True


def create_vec_list(n):
    vectors = []
    counter = -1
    for i in range(n):
        vectors.append(throw())
        if not distance(vectors):
            break
        counter += 1
    return counter


def monte_carlo(n):
    vectors = []
    scores = []
    for i in range(n):
        scores.append(create_vec_list(1000))
    return scores


def expected_value(scores):
    mean = sum(scores) / float(len(scores))
    return mean


def prob_more_than_1(scores):
    counter = 0
    for i in scores:
        if i > 1:
            counter += 1
        else:
            continue
    return counter / float(len(scores))


def plot_throws():
    ax = plt.subplot(111, projection='polar')
    ax.grid(False)
    ax.set_rmax(1)
    ax.set_yticklabels([])
    # plt.ion

    scores = []

    for iteration in range(1000):
        vectors = []

        for throws in range(50):  ##make 50 possible vector throws (won't need more than 7) --exhaustive for emphasis
            vectors.append(throw())
        vectors_plotted = []  ##track vectors that have been plotted
        score = -1  ##-1 indexed to match the problem statement
        best = 0
        mean = 0
        if iteration != 0:
            best = max(scores)
            mean = sum(scores) / float(len(scores))

        for vector in vectors:

            vectors_plotted.append(vector)
            ax.grid(False)
            ax.set_rmax(1)
            ax.set_yticklabels([])
            annotation_string = "Mean: " + str(mean) + ', Best: ' + str(best) + ", Iteration: " + str(iteration + 1)
            plt.title(annotation_string)

            if distance(vectors_plotted):
                ax.plot(vector[1], vector[0], "x", ms=10, color='green')
            plt.pause(0.25)
            if not distance(vectors_plotted):
                ax.plot(vector[1], vector[0], "x", ms=10, color='red')
                plt.pause(0.25)
                scores.append(score)
                break

            score += 1

        plt.cla()
    while True:
        plt.pause(0.1)


# plot_throws()



def separation(vector, target):
    # modulus = math.sqrt(vector[0]**2 + target[0]**2 -2*vector[0]*target[0]*math.cos(vector[1]-target[1]))
    # displacement = [vector[0] * math.cos(vector[1]) - target[0] * math.cos(target[1]), vector[0] * math.sin(vector[1]) - target[0] * math.sin(target[1])]
    # modulus = math.sqrt(float(displacement[0] ** 2 + displacement[1] ** 2))
    modulus = math.sqrt((vector[0] - target[0]) ** 2 + (vector[1] - target[1]) ** 2)
    return modulus


def dis_to_wall(vector, endpoints):
    # print endpoints[0], endpoints[1]
    line = subtract(endpoints[0], endpoints[1])

    v = subtract(vector, endpoints[0])
    u = subtract(vector, endpoints[1])

    component = -dot_product(v, line) / float(norm(line)) ** 2
   # p = endpoints[0] + scalmul(component, line)
    projection = subtract(scalmul(dot_product(v, line) / float(norm(line))**2, line), scalmul(-1,endpoints[0]))

    if component >= 0 and component <= 1:

        minimum = norm(subtract(vector, projection))
    elif component < 0:

        minimum = norm(v)
    elif component > 1:

        minimum = norm(u)

    # line = [endpoints[1][0]-endpoints[0][0], endpoints[1][1]-endpoints[0][1]]
    # projection = scalmul(dot_product(vector, line) / float(norm(line)), line)
    # dis1 = norm(subtract(vector, endpoints[0]))
    # dis2 = norm(subtract(vector, endpoints[1]))
    # dis3 = norm(subtract(vector, projection))
    # minimum = min(dis1,dis2,dis3)


    a = -1
    if norm(projection) < 0:
        a = 1
    dis_to = [minimum, a]

    return dis_to

def jump(vector, lines):
    random_jump = subtract(vector, scalmul(-0.1, throw()))
    while any(dis_to_wall(vector, line) < 0.05 for line in lines):
        random_jump = subtract(vector, scalmul(-0.1, throw()))
    return random_jump


def grad_decent(vector, target, lines, iteration):
    step_size = 0.05
    distances = {}
    vectorin = vector

    for eX in np.arange(-1, 1, 0.01):
        for eY in np.arange(-1, 1, 0.01):
            distances[(eX, eY)] = separation([vector[0] + eX, vector[1] + eY], target)


    num_close = 1
    away_from_line = []
    for i in lines:

        for j in lines:
            if dis_to_wall(vector,i)[0] < 0.05 and dis_to_wall(vector,j)[0]  < 0.05 and i != j:
                print "CORNER IDENTIFIED"
                iline = subtract(i[1],i[0])
                jline = subtract(j[1],j[0])
                away_from_line = subtract(scalmul(-1, iline),jline)
                #print away_from_line
                num_close +=1
    print num_close

    #too close to a single wall
    if any(dis_to_wall(vector, line)[0] < 0.05 for line in lines):
        distances = {}
        counter = 0
        for line in lines:
            distances[counter] = dis_to_wall(vector,line)
            counter += 1

        which = min(distances, key=distances.get)
        line = lines[which]

        along_line = subtract(line[1], line[0])
        vector = subtract(vector, scalmul(dis_to_wall(vector, line)[1] * 0.05, along_line))
        print "WALL"


    #jump away from both walls if in a corner
    elif num_close > 1:
        print "CORNER"
        vector = subtract(vector, scalmul(dis_to_wall(vector, line)[1] * -0.05, away_from_line ))


   # elif subtract(vectorin,vector) < 0.01:
   #     vector = jump(vector, lines)

    else:

        #print "WE ARE WRONG: " + str(iteration)
        [eX, eY] = min(distances, key=distances.get)
        vector = [vector[0] + eX * step_size, vector[1] + eY * step_size]


    return vector


def instance():
    lines = []

    lines = [[[-1,0],[1,0]], [[0,1],[0,-1]]]
    # for i in range(3):
    #     points = [throw(), throw()]
    #     lines.append(points)
    #     print len(lines)

    target = throw()
    vector = throw()
    ax = plt.subplot(111)


    # ax.grid(False)
    # ax.set_rmax(1)
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])


    #lines.append(points)
    ax.plot(vector[0], vector[1], "x", color="green")
    ax.plot(target[0], target[1], "x", color="red")
    lc = mc.LineCollection(lines)
    ax.add_collection(lc)
    ax.set_autoscale_on(False)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.pause(0.01)

    # vector = grad_decent(vector, target)
    # ax.plot(vector[1],vector[0], "x", color="green")
    # ax.plot(target[1],target[0], "x", color="red")
    # plt.pause(1)

    iteration = 1
    while separation(vector, target) > 0.01:
        plt.title("Iteration: " + str(iteration))
        iteration += 1
        ax.plot(vector[0], vector[1], "x", color="green")
        plt.pause(0.1)

        vector = grad_decent(vector, target, lines, iteration)
    plt.close()

    print "WE MADE IT!"


instance()



ASDASDASDASDASDASD
