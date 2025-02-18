load("Niek_RDW_skidpad_7.3ms_2024_11_23T145500_export.mat");
% [t10, t11] = load("Niek_RDW_skidpad_flatout_2024_11_23T145959_export.mat");

t = t0.time;
t = t(1):0.004:t(end);

r = interp1(t0.time, t0.ControlsOut_GyroZ, t);
steer = interp1(t0.time, t0.ControlsOut_SteeringSteer, t);
vx = interp1(t1.time, t1.velx, t);
vy = interp1(t1.time, t1.vely, t);

t = t(1):0.004:t(end);

sys = lpvss('vx', @dataFcnSteering);
% sys = c2d(sys, 0.004, 'tustin');

[t2, r2, vy2, vx2, steer2] = preprocess(t, r, vy, vx, steer);

[y2, tOut] = lsim(sys, steer2, t2, [0, 0, 0], vx2);

plots = true;
if plots
    figure;
    hold on
    plot(y2(:, 1))
    plot(-vy2)
    legend("sim", "gss")
    title("vy")
    hold off
    
    figure;
    hold on
    plot(y2(:, 2))
    plot(-r2)
    legend("sim", "gss")
    title("r")
    hold off
end 

[A_10, B_10, C_10, D_10, ~] = dataFcnSteering(0, 10);

tune_sys_10 = ss(A_10, B_10, C_10, D_10);

yawrate_10 = c2d(tune_sys_10(2), 0.004, 'foh');
% yawrate_10.InputDelay = 20;


[A_5, B_5, C_5, D_5, ~] = dataFcnSteering(0, 5);

tune_sys_5 = ss(A_5, B_5, C_5, D_5);

yawrate_5 = c2d(tune_sys_5(2), 0.004, 'foh');
% yawrate_5.InputDelay = 20;

lqr_qs = [5, 0.1, 5];
lqr_rs = 0.1;
lqr_cost = diag(1./lqr_qs./lqr_qs);
lqr_n = zeros(3, 1);
lqr_r = 1/lqr_rs/lqr_rs;

gains_5 = dlqr(tune_sys_5.A, tune_sys_5.B, lqr_cost, lqr_r, lqr_n)
gains_10 = dlqr(tune_sys_10.A, tune_sys_10.B, lqr_cost, lqr_r, lqr_n)



% states: vy, r, heading
function [A,B,C,D,E,dx0,x0,u0,y0,Delays] = dataFcnSteering(~,vx)
    m = 220;
    Iz = 300;
    C_data_y = [1.537405752168591e+04, 2.417765976460659e+04, 3.121158998819641e+04, 3.636055041362088e+04];
    C_data_x = [300 500 700 900];
    wheelbase = 1.53;
    lr = wheelbase*(1-0.51);
    lf = wheelbase - lr;
    % [front, rear]
    C = [interp1(C_data_x, C_data_y, (9.81*m/2)*(lr/wheelbase))*2 interp1(C_data_x, C_data_y, (9.81*m/2)*(lf/wheelbase))*2];
    L = [lf lr];

    steering_scaling = 0.4 /(pi/2);

    A = [-(C(1) + C(2)) / (m * vx), vx + (C(2)*L(2) - C(1)*L(1)) / (m * vx), 0;
        (C(2)*L(2) - C(1)*L(1)) / Iz, -(L(1)*L(1)*C(1) + L(2)*L(2)*C(2)) / (Iz * vx), 0;
        0 1 0];

    B = [-C(1) / m; -(L(1) * C(1)) / Iz; 0] * steering_scaling;

    C = eye(3);
    D = [0; 0; 0];           
    E = [];
    
    dx0 = [0; 0; 0];
    x0 = [0;0;0];
    u0 = [0];
    y0 = [0;0;0];
    Delays = [];
end


function[t, r, vy, vx, steer] = preprocess(t, r, vy, vx, steer)
    start_index = find(vx>0.01, 1, "first");
    t = t(start_index:end);
    r = r(start_index:end);
    vy = vy(start_index:end);
    vx = vx(start_index:end);
    steer = steer(start_index:end);

    end_index = find(vx<0.01, 1, "first");
    t = t(1:end_index);
    r = r(1:end_index);
    vy = vy(1:end_index);
    vx = vx(1:end_index);
    steer = steer(1:end_index);
end