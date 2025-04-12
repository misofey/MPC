import numpy as np

def check_control_amissible_invariance(P, f:callable, K, c) -> bool:
    
    # Obtain diagonal form of P
    eigvals, eigvecs = np.linalg.eigh(P)
    Q = eigvecs 
    
    # Bounds of outer axis parallel polyhadron in the derotated coordinate frame
    bounds = np.sqrt(c / eigvals)

    # Generate vertices of the outer axis parallel polyhedron in the derotated coordinate frame
    vertices = np.array(np.meshgrid(*[[-b, b] for b in bounds])).T.reshape(-1, len(bounds))
    print(vertices.shape)
    print(2**(len(bounds)))

    for vertex in vertices:
        # Transform back to original coordinate frame
        x = Q @ vertex
        # Apply the control law
        x_next = f(x - K @ x)
        # Transform back to derotated coordinate frame
        vertex_next = Q.T @ x_next
        # Check if the vertex is in the proposed invariant set
        if np.any(np.abs(vertex_next) > bounds):
            return False
        
    return True