import numpy as np

def check_control_amissible_invariance(P, f:callable, K, c) -> bool:
    n = P.shape[0]
    eigvals, eigvecs = np.linalg.eigh(P)
    Q = eigvecs  # columns are eigenvectors
    
    bounds = np.sqrt(c / eigvals)

    vertices = np.array(np.meshgrid(*[[-b, b] for b in bounds])).T.reshape(-1, len(bounds))
    print(vertices.shape)
    print(2**(len(bounds)))

    for vertex in vertices:
        # Transform to principal axes frame
        x_next = f(vertex - K @ vertex)
        print(x_next)
        x_next_rot = Q @ x_next
        print((Q @ vertex)-K @ (Q @ vertex))
        # Check if the vertex is in the invariant set
        if np.any(np.abs(x_next_rot) > bounds):
            return False
        
    return True