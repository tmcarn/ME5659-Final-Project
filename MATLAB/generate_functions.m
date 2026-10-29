%%%% DCG matrices
set_path

syms x theta dx dtheta u real
syms m M l g real

% Define parameters in a vector
params = [m, M, l, g];

q = [x; theta];
dq = [dx; dtheta];

p_cm = [x + l*sin(theta); l*cos(theta)];
dp_cm = jacobian(p_cm, q) * dq;

K_cart = (1/2) * M * dx^2;
K_pend = (1/2) * m * (dp_cm.' * dp_cm);
K = K_cart + K_pend;
K = simplify(K);

V = m * g * l * (1 - cos(theta));


% Mass Inertial Matrix
D = jacobian(jacobian(K,dq).',dq); 
D = simplify(D)

% Coriolis matrix
N = max(size(q));
syms C
for k = 1:N
    for j = 1:N
        C(k,j) = 0*g;
        for i = 1:N
            C(k,j) = C(k,j) + 1/2*(diff(D(k,j),q(i)) + ...
                diff(D(k,i),q(j)) - ...
                diff(D(i,j),q(k)))*dq(i);
        end
    end
end
C = simplify(C)

% Gravity matrix
G = jacobian(V,q).'; 
G = simplify(G)

B = [1;0];

write_symbolic_term_to_mfile(q,dq,params,D,C,G)

%% ---- Linearize to State-Space ----
% Equations of motion: D*ddq + C*dq + G = B*u

% Equilibrium Point:
eq_point = [q; dq];
eq_vals = [0; 0; 0; 0];

% Solve for ddq using Inverse Dynamics
ddq = D \ (B*u - C*dq - G);
ddq = simplify(ddq);

% First-order state form: state = [q; dq], dstate = [dq; ddq]
state = [q; dq];
dstate = [dq; ddq];

% Jacobian linearization
A_sym = jacobian(dstate, state);
B_sym = jacobian(dstate, u);

% Evaluate at equilibrium, (zero velocity, zero input)
A0 = subs(A_sym, [state; u], [eq_vals; 0]);
B0 = subs(B_sym, [state; u], [eq_vals; 0]);

A0 = simplify(A0)
B0 = simplify(B0)

write_symbolic_term_to_mfile(q,dq,params, A0, B0)


