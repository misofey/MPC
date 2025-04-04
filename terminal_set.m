addpath(genpath('./tbxmanager'))
mpt_init

A = [1.          0.3         0.02        0.          0.        ;
 0.          1.          0.          0.02        0.        ;
 0.          0.          0.6802073  -0.00369231 -2.0977733 ;
 0.          0.         -0.00219257  0.8871941  -1.1200825 ;
 0.          0.          0.          0.          1.        ];
B =  [0.; 0.; 0.; 0.; 0.02];
dt = 0.02;

model = LTISystem('A', A, 'B', B);

model.x.min = [-inf -inf -inf -inf -.8];
model.x.max = [inf inf inf inf  .8];
model.u.min = -1.6;
model.u.max = 1.6;


X_constraints = Polyhedron('lb', [-inf -inf -inf -inf -0.8], 'ub', [inf inf inf inf 0.8]);
% Define input constraints as a Polyhedron
U_constraints = Polyhedron('lb', -1.6, 'ub', 1.6);

ops = {};
ops.maxIterations = 26;  % Set maximum iterations
mptopt('lpsolver', 'GLPK');      % Use GLPK (works well for many cases)
mptopt('qpsolver', 'quadprog');  % MATLAB’s built-in QP solver (if QP is used)

% Compute the invariant set with custom options
%X_pre = model.reachableSet('X', X_constraints, 'U', U_constraints, 'direction', 'backward', 'N', 5);
o = model.invariantSet('X', X_constraints, 'U', U_constraints, 'maxIterations', 26);

% Plot the invariant set if it's computed
% Project the invariant set to the first three dimensions (x1, x2, x3)
o3D = X_pre.projection([2, 4]);

% Plot the 3D projection
figure;
o3D.plot('color', 'b');
grid on;

o.Dim

save('invariant_set.mat', 'o');