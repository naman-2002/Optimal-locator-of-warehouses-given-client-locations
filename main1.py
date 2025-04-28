from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

points = []  # Initialize an empty list to store points

@app.route('/process_points', methods=['POST'])
def process_points():
    data = request.json  # Get the JSON data from the request
    received_points = data.get('points', [])  # Extract the 'points' array from the JSON data
    p_value = data.get('p')
    points.extend(received_points)  # Add received points to the global 'points' list
    M = received_points
    print(M)
    m = len(M)
    p = int(p_value)

    from scipy.optimize import linprog
    import numpy as np
    from sklearn.cluster import KMeans
    from typing import List, Tuple

    print("The value of m is: ", m)
    # p = int(input("Total no. of store house need: "))
    print("The value of p is: ", p)


    from scipy.optimize import linprog
    import numpy as np
    from sklearn.cluster import KMeans
    from typing import List, Tuple

    # m = len(M)
    # print("The value of m is: ", m)
    # p = int(input("Total no. of store house need: "))
    # print("The value of p is: ", p)


    def format_function(M: List[Tuple[float, float]], p: int) -> list[float]:  # objective function
        c = [0 for j in range(2 * p + 2 * len(M) + 3 * len(M) * p)]
        st = 2 * p + 2 * len(M)
        en = st + len(M) * p
        for i in range(st, en):
            c[i] = 1
        return c


    c = format_function(M, p)
    # print("c: ", c)


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
            # consider the next (xi,yi) when the last element is reached
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
    # print("A_ub: ", A_ub)


    # print(len(A_ub))


    def format2_function(M: List[Tuple[float, float]], p: int) -> list[float]:
        b_ub = [0 for j in range(5 * p * len(M))]
        return b_ub


    b_ub = format2_function(M, p)
    # print("b_ub: ", b_ub)


    # now defining the given equation

    def format3_function(M: List[Tuple[float, float]], p: int) -> list[list]:
        A_eq = [[0 for b in range(2 * p + 2 * len(M) + 3 * len(M) * p)] for k in range(2 * len(M))]

        st1 = 2 * p
        en1 = st1 + 2 * len(M)
        for i in range(2 * len(M)):
            A_eq[i][i + 2 * p] = 1

        return A_eq


    A_eq = format3_function(M, p)

    # print("A_eq: ", A_eq)


    # b_eq.extend([x1, y1, x2, y2])

    # x1 = M[0][0]
    # x2 = M[0][1]

    # print(x1, x2)

    def format4_function(M: List[Tuple[float, float]], p: int) -> list[float]:
        b_eq = [0 for j in range(2 * len(M))]
        for i in range(len(M)):
            for j in range(2):
                b_eq[2 * i + j] = M[i][j]

        return b_eq


    b_eq = format4_function(M, p)

    # print("b_eq: ", b_eq)


    def compute_simplex(M: List[Tuple[float, float]], p: int) -> Tuple[float, float]:
        c = format_function(M, p)
        A_ub = format1_function(M, p)
        b_ub = format2_function(M, p)
        A_eq = format3_function(M, p)
        b_eq = format4_function(M, p)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        # adjust 0f in nf for n decimals precision
        return float("%.0f" % res.x[0]), float("%.0f" % res.x[1])


    b = compute_simplex(M, p)
    print(b)


    # res = linprog(C, A_ub = A_ub, b_ub= b_ub, A_eq= A_eq, b_eq= b_eq)
    #
    # print(res.message)
    #
    # x = res.x
    #
    # print(x)
    # print(float(x[0]))
    # print(float(x[1]))


    def cluster(M: List[Tuple[float, float]], p: int) -> List[List]:
        data = np.vstack(M)
        kmeans = KMeans(n_clusters=p)
        label = kmeans.fit_predict(data)  # we provide the indexes of cluster to each data points
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
        # Original list of tuples
    # list_of_tuples = [(1, 2), (3, 4), (5, 6)]

    # Convert list of tuples to list of lists
    g = [list(t) for t in f]

    print(g)

    return (g)

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode

