import random
import numpy as np

#  Python 3.9.4
ADABOOST_ITERATIONS = 100


class Point:
    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.weight = 0.0
        self.classification = c
        self.final_prediction = []


class Rule:
    def __init__(self, x1, y1, x2, y2, te=0.0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.total_error = te
        self.alpha = 0.0
        self.plus_points = []
        self.minus_points = []


def get_rule_prediction(a, b, c, x0, y0, flag):
    if b > 0:
        a = -a
        b = -b
        c = -c
    temp = a * x0 + b * y0 + c
    # flag indicates the direction of the line
    if flag:
        return temp <= 0
    else:
        return temp >= 0


def line_from_points(p1, q):
    a = q[1] - p1[1]
    b = p1[0] - q[0]
    c = -(a * (p1[0]) + b * (p1[1]))
    return a, b, c


def adaboost(rules_set, points_set):
    best_rule = Rule(0, 0, 0, 0, te=1)
    for ij in range(len(rules_set)):  # loop each rule
        rules_set[ij].total_error = 0.0
        for point in points_set:  # loop each point for each rule
            # if rule predicts + but point actually - or rule predicts - but point actually +
            if (point in rules_set[ij].plus_points and point.classification == -1) or \
                    (point not in rules_set[ij].plus_points and point.classification == 1):
                rules_set[ij].total_error += point.weight
        # compute rule`s alpha
        rules_set[ij].alpha = 0.5 * np.log((1 - rules_set[ij].total_error) / rules_set[ij].total_error)
        # keep rule with lowest weighted error
        if rules_set[ij].total_error < best_rule.total_error:
            best_rule = rules_set[ij]
    return best_rule


def update_weights(best_rule, points):
    weights_sum = 0
    for u in range(len(points)):
        if points[u].x == best_rule.x1 and points[u].y == best_rule.y1 or \
                points[u].x == best_rule.x2 and points[u].y == best_rule.y2:
            continue
        # two cases of misclassification
        if (points[u] in best_rule.plus_points and points[u].classification == -1) or \
                (points[u] not in best_rule.plus_points and points[u].classification == 1):
            points[u].weight = points[u].weight * np.exp(best_rule.alpha)
        else:
            points[u].weight = points[u].weight * np.exp(-1 * best_rule.alpha)

        weights_sum += points[u].weight
    for u in range(len(points)):
        points[u].weight = points[u].weight / weights_sum


if __name__ == '__main__':

    # read file
    f = open("rectangle.txt")
    read_lines = f.read().splitlines()

    # split lines into array '1.25 2.50 1' --> [1.25, 2.50, 1]
    points_list = []
    for i in range(len(read_lines)):
        points_list.insert(i, read_lines[i].split(' '))

    # create array containing all Points
    points_arr = []
    for i in range(len(points_list)):
        points_arr.insert(i, Point(float(points_list[i][0]), float(points_list[i][1]), int(points_list[i][2])))
    # avoid divide by zero exception
    np.seterr(divide='ignore')

    # create array containing all possible lines (rules)
    rules = []
    for i in range(len(points_list)):
        for j in range(0, len(points_list)):
            if i != j:
                rules.insert(i,
                             Rule(float(points_list[i][0]), float(points_list[i][1]),
                                  float(points_list[j][0]), float(points_list[j][1])))
    #  set for every rule the set of points tagged as +
    for r in rules:  # loop each rule
        a1, b1, c1 = line_from_points([r.x1, r.y1], [r.x2, r.y2])
        for p in points_arr:  # loop each point for each rule
            if get_rule_prediction(a1, b1, c1, p.x, p.y, np.sqrt(r.x1**2 + r.y1**2) < np.sqrt(r.x2**2 + r.y2**2)):
                r.plus_points.append(p)
            else:
                r.minus_points.append(p)

    true_error, empirical_error = [0 for _ in range(8)], [0 for _ in range(8)]
    for _ in range(ADABOOST_ITERATIONS):

        # split data randomly into two halves
        random.shuffle(points_arr)
        arr_size = int(len(points_arr) / 2)
        train_data = points_arr[:arr_size]
        test_data = points_arr[arr_size:]

        # init points weights
        for ll in train_data:
            ll.weight = 1 / len(train_data)

        # execute 8 iteration of adaboost to find the 8 best rules (chosen by the biggest alpha)
        results = []
        for i in range(8):
            result = adaboost(rules, train_data)
            results.insert(i, [result, result.alpha])
            update_weights(result, train_data)

        # build Hk(x) rules
        Hk = [[results[0]],
              [results[0], results[1]],
              [results[0], results[1], results[2]],
              [results[0], results[1], results[2], results[3]],
              [results[0], results[1], results[2], results[3], results[4]],
              [results[0], results[1], results[2], results[3], results[4], results[5]],
              [results[0], results[1], results[2], results[3], results[4], results[5], results[6]],
              [results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]]]

        # get Hk(x) final prediction on every point
        for p in points_arr:
            p.final_prediction = []
            for i in range(len(Hk)):
                final_prediction = 0.0
                for j in range(len(Hk[i])):
                    if p in Hk[i][j][0].plus_points:
                        prediction = 1
                    else:
                        prediction = -1
                    final_prediction += Hk[i][j][1] * prediction
                p.final_prediction.insert(i, np.sign(final_prediction))

        # compute empirical error of every Hk(x)
        # empirical error  = num of misclassified points in train set / num of points
        for i in range(8):
            counter = 0
            for p2 in train_data:
                if (p2.classification == 1 and p2.final_prediction[i] == -1) or \
                        (p2.classification == -1 and p2.final_prediction[i] == 1):
                    counter += 1
            empirical_error[i] += counter / len(train_data)

        # compute true error of every Hk(x)
        # true error  = num of misclassified points in test set / num of points
        for i in range(8):
            counter = 0
            for p2 in test_data:
                if (p2.classification == 1 and p2.final_prediction[i] == -1) or \
                        (p2.classification == -1 and p2.final_prediction[i] == 1):  # two cases of misclassification
                    counter += 1
            true_error[i] += counter / len(test_data)

    for k in range(8):
        print("true_error avg for ", ADABOOST_ITERATIONS, " runs: ", true_error[k] / ADABOOST_ITERATIONS,
              ", empirical_error: ", empirical_error[k] / ADABOOST_ITERATIONS)

