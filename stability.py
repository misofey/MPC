import numpy as np
from matplotlib import pyplot as plt
import cvxpy as cp

def estimate_control_amissible_invariant_set(K, A, B, ubx, ubu, verbose=False) -> float:

    original_constraints = []
    phi = A - B@K

    print(f"phi eigen vlaues: {np.linalg.eigvals(phi)}")
    # [A - BK] @ x = phi @ x = [[x_k+1]
    # [  BK  ]                  [  u ]]
    condition = True

    n = 1

    n_max = 0

    constraints = []
    
    x = cp.Variable((5, 1))

    constraints.append(cp.abs(x) <=  ubx.reshape(-1, 1))
    constraints.append(cp.abs(K@x) <= ubu.reshape(-1, 1))
    opt_vlaues = []

    while condition:
        print(f"--- Iteration {n} ---")
        phi_np1 = np.linalg.matrix_power(phi, n)
        constraints.append(cp.abs(phi_np1 @ x) <=  ubx.reshape(-1, 1))
        constraints.append(cp.abs(K @ (phi_np1 @ x)) <=  ubu.reshape(-1, 1))
        n += 1
        phi_np1 = np.linalg.matrix_power(phi, n)
        print(f"phi_np1: {phi_np1}")
        
        try:

            for j in range(len(ubx)):
                cost = cp.Maximize((phi_np1 @ x)[j]/ubx[j])
                problem = cp.Problem(cost, constraints)
                problem.solve(solver = cp.ECOS, verbose=verbose)
                opt_value = problem.value
                print(f"Solution found: {opt_value}")
                opt_vlaues.append(opt_value)
        except cp.error.SolverError:
            print("Solver error: Problem is infeasible")
        try:    
            for j in range(len(ubu)):
                cost = cp.Maximize((- K @ (phi_np1 @ x))[j]/ubu[j]) # check this
                problem = cp.Problem(cost, constraints)
                problem.solve(solver = cp.ECOS, verbose=verbose)
                opt_vlaues.append(problem.value)
        except cp.error.SolverError:
            print("Solver error: Problem is infeasible")

        #print(f"opt_vlaues: {opt_vlaues}")
        if np.all(np.array(opt_vlaues) < 1e-5):
            condition = False
            print(f"n: {n}")
            n_max = n
        if n > 100:
            print("n is too large")
            condition = False
            


def check_control_amissible_invariance(P, f:callable, c) -> bool:
    
    # Obtain diagonal form of P
    eigvals, eigvecs = np.linalg.eigh(P)
    Q = eigvecs 
    # P = Q @ np.diag(eigvals) @ Q.T
    # x = Q @ y => y = Q.T @ x

    # Bounds of outer axis parallel polyhadron in the derotated coordinate frame
    bounds = np.sqrt(c / eigvals)
    print(f"bounds: {bounds}")
    # Generate vertices of the outer axis parallel polyhedron in the derotated coordinate frame
    vertices = np.array(np.meshgrid(*[[-b, b] for b in bounds])).T.reshape(-1, len(bounds))
    #print(f"vertices: {vertices}")
    print(f"shape of vertex matrix: {vertices.shape}")
    print(2**(len(bounds)))
   
    for (i, vertex) in enumerate(vertices):
        # Transform back to original coordinate frame
        x = Q @ vertex
        # Apply the control law
        x_next = f(x)
        #print(f"x: {x}")
        #print(f"x_next: {x_next}")
        # Transform back to derotated coordinate frame
        vertex_next = Q.T @ x_next
        print(f"vertex: {vertex}")
        print(f"vertex_next: {vertex_next}")
        # Check if the vertex is in the proposed invariant set
        print(np.abs(vertex_next) < bounds)
        if np.any(np.abs(vertex_next) > bounds):
            print(f"Non control amissible ivaariant for c = {c}")
            return False
        
    print(f"Control amissible ivariant for c = {c}")
    return True

def binary_search(P, f:callable, c_u, epsilon=1e-1) -> float:
    #c = np.linspace(10e-50, c_u, 1000000000)
    #for c_i in c:
    #    if check_control_amissible_invariance(P, f, c_i):
    #        c_best = c_i
    #        break
    c_l = 0
    c = 0
    c_best = 0
    i = 0
    if check_control_amissible_invariance(P, f, c_u):
        print(f"initial guess is control amissible ivaariant for c = {c_u}")
        return c_u
    while c_u - c_l > epsilon:
        print(f"---- Iteration {i} ----")
        c = (c_l + c_u) / 2
        if check_control_amissible_invariance(P, f, c):
            c_best = c
            c_l = c
        else:
            c_u = c
        i+= 1
    print(f"c_best: {c_best}")
    return c_best