%% Residual
function [R] = residual_constrained(X, Y, F, G, N, n_x, n_y, std_w, std_v)
    % Pass h in last row.
    h = X(end);
    X = X(1:size(X,1)-1, 1);
    
    % Assuming X is a long vector, get the new indices.
    x_current = X(1:n_x, 1);
    n1 = n_x + 1;
    n2 = 2*n_x;
    x_next = X(n1:n2, 1);
    
    R = Y(:, 1) - G(x_current);
    
    W = (1./std_v).*eye(N*n_y);
    %W = [eye(N*n_x)  zeros(N*n_x, N*n_y);
    %     zeros(N*n_y, N*n_x)    (1./std_v).*eye(N*n_y)];
    for n=2:N
        x_current = x_next;
        
        % Assuming X is a long vector, get the new indices.
        n1 = n*n_x + 1;
        n2 = (n+1)*n_x;
        x_next = X(n1:n2, 1);
        
        % Computing the resisduals.
        R = [R; Y(:, n) - G(x_current)];
    end
    R = W*R;
end
