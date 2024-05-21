from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from typing import List, Tuple

aWeb_1 = Flask(__name__)
CORS(aWeb_1)

points = []  # Initialize an empty list to store points

@aWeb_1.route('/process_points', methods=['POST'])
def process_points():
    print("Received a request on /process_points")
    data = request.json  # Get the JSON data from the request
    if data is None:
        print("No JSON data received")
        return jsonify({"error": "No JSON data received"}), 400

    received_points = data.get('points', [])  # Extract the 'points' array from the JSON data
    p_value = data.get('p')
    if not received_points or p_value is None:
        print("Missing 'points' or 'p' in the request")
        return jsonify({"error": "Missing 'points' or 'p' in the request"}), 400

    points.extend(received_points)  # Add received points to the global 'points' list
    M = received_points
    m = len(M)
    p = int(p_value)

    print("The value of m is: ", m)
    print("The value of p is: ", p)

    def format_function(M: List[Tuple[float, float]], p: int) -> List[float]:
        c = [0 for j in range(2 * p + 2 * len(M) + 3 * len(M) * p)]
        st = 2 * p + 2 * len(M)
        en = st + len(M) * p
        for i in range(st, en):
            c[i] = 1
        return c

    c = format_function(M, p)

    def format1_function(M: List[Tuple[float, float]], p: int) -> List[List[int]]:
        A = [[0 for i in range(len(M) * (2 + 3 * p) + p * 2)] for _ in range(len(M) * 5 * p)]
        x = 0
        y = 1
        d = p * 2 + len(M) * 2
        a = p * 2
        b = p * 2 + 1
        dx = p * 2 + len(M) * (2 + p)
        dy = p * 2 + len(M) * (2 + p) + 1
        it = 0
        for i in range(0, len(M) * p):
            # constraint 1
            A[it][dx] = 1
            A[it][dy] = 1
            A[it][d] = -1
            # constraint 2
            A[it + 1][x] = 1
            A[it + 1][a] = -1
            A[it + 1][dx] = -1
            # constraint 3
            A[it + 2][x] = -1
            A[it + 2][a] = 1
            A[it + 2][dx] = -1
            # constraint 4
            A[it + 3][y] = 1
            A[it + 3][b] = -1
            A[it + 3][dy] = -1
            # constraint 5
            A[it + 4][y] = -1
            A[it + 4][b] = 1
            A[it + 4][dy] = -1
            # update indexes
            if i % len(M) == len(M) - 1:
                x += 2
                y += 2
                d = p * 2 + len(M) * 2
                a = p * 2
                b = p * 2 + 1
            else:
                a += 2
                b += 2
                d += 1
            dx += 2
            dy += 2
            it += 5
        return A

    A_ub = format1_function(M, p)

    def format2_function(M: List[Tuple[float, float]], p: int) -> List[float]:
        b_ub = [0 for j in range(5 * p * len(M))]
        return b_ub

    b_ub = format2_function(M, p)

    def format3_function(M: List[Tuple[float, float]], p: int) -> List[List[int]]:
        A_eq = [[0 for b in range(2 * p + 2 * len(M) + 3 * len(M) * p)] for k in range(2 * len(M))]

        st1 = 2 * p
        en1 = st1 + 2 * len(M)
        for i in range(2 * len(M)):
            A_eq[i][i + 2 * p] = 1

        return A_eq

    A_eq = format3_function(M, p)

    def format4_function(M: List[Tuple[float, float]], p: int) -> List[float]:
        b_eq = [0 for j in range(2 * len(M))]
        for i in range(len(M)):
            for j in range(2):
                b_eq[2 * i + j] = M[i][j]

        return b_eq

    b_eq = format4_function(M, p)

    def compute_simplex(M: List[Tuple[float, float]], p: int) -> Tuple[float, float]:
        c = format_function(M, p)
        A_ub = format1_function(M, p)
        b_ub = format2_function(M, p)
        A_eq = format3_function(M, p)
        b_eq = format4_function(M, p)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        return float("%.0f" % res.x[0]), float("%.0f" % res.x[1])

    b = compute_simplex(M, p)
    print(b)

    def cluster(M: List[Tuple[float, float]], p: int) -> List[List]:
        data = np.vstack(M)
        kmeans = KMeans(n_clusters=p)
        label = kmeans.fit_predict(data)
        clus = [[] for _ in range(p)]
        for i in range(m):
            index = label[i]
            clus[index].append((data[i][0], data[i][1]))
        return clus

    a = cluster(M, p)
    print(a)

    def solve(M: List[Tuple[float, float]], p: int) -> List[Tuple[float, float]]:
        clusters = cluster(M, p)
        results = []
        for t in clusters:
            results.append(compute_simplex(t, 1))
        return results

    f = solve(M, p)
    g = [list(t) for t in f]

    print(g)

    return jsonify(g)

if __name__ == '__main__':
    aWeb_1.run(host='0.0.0.0', port=8000)

