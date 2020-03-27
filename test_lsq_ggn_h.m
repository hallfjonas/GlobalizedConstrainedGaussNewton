close all;
clear;

h_init = 0.05; %For now.

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
R_no_const = @(X) residual(X, Y, F, G, N, n_x, n_y, std_w, std_v);

%% Initial guess: simulate system deterministically
x_init_true = [5 -7 1.7 5 5 10]' + 1;  % true inital state

X_init = zeros(n_x, N+1);
X_init(:, 1) = x_init_true;

for n=1:N
    X_init(:, n+1) = F([X_init(:, n); h_init]);
end

X_init = reshape(X_init, (N+1)*n_x, 1);    % reshape to column vector
X_init = [X_init; h_init];


%% Solve estimation problem

tic;
[X_opt_col_lsq, ~, R_opt_lsq, ~, ~, ~, J] = lsqnonlin(R_no_const, X_init);
time_lsq = toc;

% Parameters
params.gamma  = 0.8;
params.delta  = 2/5;
params.x_0    = X_init;
params.mu_bar = [1, 2];

G_no_const = @(x) 0;

tic;
[X_opt_col, X_all_col, R_opt, J, R_norm_all, G_norm_all] = global_gauss_newton(R_no_const, G_no_const, params);
time_ggn = toc;

fprintf('norm(R_opt_lsq) - norm(R_opt) = %e || time_lsq = %f s || time_ggn = %f s\n', norm(R_opt_lsq) - norm(R_opt), time_lsq, time_ggn);
