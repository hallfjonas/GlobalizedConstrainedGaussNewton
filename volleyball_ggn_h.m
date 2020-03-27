close all;
clear;

% Import envoironments.
import casadi.*

%% Load measurements

data = load('volleyball_cameras.mat');

Y = data.Y_measured;   % measurements, n_y x N              [1, ..., N]
X_true = data.X_true;  % true state trajectory, n_x x (N+1) [1, ..., N+1]

%% Parameter
N = size(Y, 2);       % Number of observations

std_w = 0.01;        % standard deviation of state noise
std_v = 5*10^(-4);   % standard deviation of measurement noise

g_true = 9.81;            % gravity
h_true = 0.05;

% Eye stuff
e_l = [1 0 0]';          % line of sight
e_u = [0 0 1]';          % up vector
e_s = cross(e_u, e_l);   % right eye to left eye
d_eyes = 1;              % dist between eyes
L = 0.02;                % dist to screen
P_R = [2  0.5 1.8]';     % position of right eye
P_L = P_R + d_eyes.*e_s; % position of left eye

% Dimensions
n_y = 4;                % two 2-dimensional cameras
n_p = 3;                % 3-dimensional position 
n_v = 3;                % 3-dimensional velocity
n_x = n_p + n_v;        % state dimension

%% State model 
A = @(h) eye(n_x) + [zeros(n_p) h.*eye(n_v);
                     zeros(n_p)  zeros(n_v)];

b = @(h) [0 0 -0.5*g_true*h^2 0 0 -h.*g_true]';

% Pass h in last row! x will be all rows up to last one.
F = @(x)    A(x(end, 1))*x(1:size(x,1) - 1, 1) + b(x(end, 1));

%% Measurement model
alpha_R = @(p) L./((p-P_R)'*e_l); 
alpha_L = @(p) L./((p-P_L)'*e_l);

u_R = @(p)  e_u'*(alpha_R(p).*(p - P_R));
v_R = @(p) -e_s'*(alpha_R(p).*(p - P_R));

u_L = @(p)  e_u'*(alpha_L(p).*(p - P_L));
v_L = @(p) -e_s'*(alpha_L(p).*(p - P_L));

G = @(x) [v_R(x(1:3,1)) u_R(x(1:3,1)) v_L(x(1:3,1)) u_L(x(1:3,1))]';

%% Residual function
%  AGAIN, H IS PASSED AS LAST ELEMENT OF X (IN THIS CASE LAST ROW?)
R_h = @(X) residual_constrained(X, Y, F, G, N, n_x, n_y, std_w, std_v);

%% Initial guess: simulate system deterministically
h_init = rand(); % Random start values.

x_init_true = [5 -7 1.7 5 5 10]' + 1;  % true inital state

X_init = zeros(n_x, N+1);
X_init(:, 1) = x_init_true;

for n=1:N
    X_init(:, n+1) = F([X_init(:, n); h_init]);
end

X_init = reshape(X_init, (N+1)*n_x, 1);    % reshape to column vector
X_init = [X_init; h_init];


%% Solve estimation problem
% Parameters
params.gamma  = 0.8;
params.delta  = 2/5;
params.x_0    = X_init;
params.mu_bar = [1, 2];

G_const = @(x) constraint(x, n_x, N, F);
[X_opt_col, X_all_col, R_opt, J, R_norm_all, G_norm_all] = global_gauss_newton(R_h, G_const, params);

% Examine h and x vals.
h_opt     = X_opt_col(end);
X_opt_col = X_opt_col(1:end-1);
h_all     = X_all_col(size(X_all_col, 1), :);
X_all_col = X_all_col(1:size(X_all_col, 1)- 1, :);
K         = length(h_all);

fprintf('h_opt = %f\n', h_opt);

% Reshape x vals.
X_opt = reshape(X_opt_col, n_x, N+1);      % n_x x (N+1) format 
X_all = zeros(n_x, N+1, K);
for k = 1:size(X_all_col,2)
    X_all(:, :, k) = reshape(X_all_col(:,k), n_x, N+1);    
end

%% Compute 2D projection of X_opt
Y_opt = zeros(n_y, N);
Y_all = zeros(n_y, N);

for n=1:N
   Y_opt(:, n) = G(X_opt(:, n));
end

for k=1:K
   for n=1:N
       Y_all(:, n, k) = G(X_all(:, n, k));
   end
end

%% Plot estimated 2D projections and measurements
figure(1); 
s1 = subplot(2,1,2); hold on; grid on; 
title('Right eye');
plot(Y(1, :), Y(2,:), 'ro');
plot(Y_opt(1, :), Y_opt(2, :), 'r.-');
legend('Y measured', 'Y estimated');

s2= subplot(2,1,1); hold on; grid on;
title('Left eye');
plot(Y(3, :), Y(4,:), 'bo');
plot(Y_opt(3, :), Y_opt(4, :), 'b.-');
legend('Y measured', 'Y estimated');

linkaxes([s1, s2], 'x')
linkaxes([s1, s2], 'y')

%% Plot h values
figure(2); hold on;
title('h values per iteration');
plot(1:K, h_all(1:K));
plot(1:K, h_true*ones(K,1));
legend('h vals', 'h true');

%% Plot x moving
figure(3); 
for i=1:K
    plot(Y(3, :), Y(4,:), 'ro', Y_all(3, :, i), Y_all(4, :, i), 'b.-');
    legend('Y measured', sprintf('Y estimated (k = %d)', i));
    anim(i) = getframe;
    pause(0.01);
end

%% Plot residual
figure(4); hold on;
title('Norm of residual per iteration');
plot(1:(K-1), R_norm_all(1:(K-1), 1));
legend('|| F_k ||');

%% Plot estimated and true 3D trajectories
figure(5); hold on; grid on;
plot3(X_true(1, :), X_true(2, :), X_true(3, :), 'kx-');
plot3(X_opt(1, :), X_opt(2, :), X_opt(3, :), 'b-');

plot3(P_R(1), P_R(2), P_R(3), 'ro', 'MarkerSize', 5); 
plot3(P_L(1), P_L(2), P_L(3), 'bo', 'MarkerSize', 5);

quiver3(P_R(1), P_R(2), P_R(3), 5*e_l(1), 5*e_l(2), 5*e_l(3), 'r')
quiver3(P_L(1), P_L(2), P_L(3), 5*e_l(1), 5*e_l(2), 5*e_l(3), 'b')

view([-100, 7]);
