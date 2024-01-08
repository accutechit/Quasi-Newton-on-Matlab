clc; clear all; close all

f = @(x) [2*x(1)+x(4)+8;
    2*x(2)+x(4)+4;
    2*x(3)+x(4);
    x(1)+x(2)+x(3)-18  
               ]
x0 = [1;1;1;1];
opt = [];
bounds = [0 10;0 10;0 10;-20 0];
[X ithist] = broyden(f,x0,opt,bounds)

Roots_Value = ithist.x;
Fx_Value = ithist.f;
Norm_F_Value = ithist.normf;
table(Roots_Value,Fx_Value,Norm_F_Value)

plot(Roots_Value,'linewidth',2); grid on
xlabel('Number of Iteration'); ylabel('Roots Convergence');
title('Broydens Method (Quasi-Newton Method)')

function [x, ithist] = broyden(f,x0,opt,bounds)
    optfields = {'maxiter','tolfun','tolx'};
    defaults = {2000,1e-10,1e-8,[]};
    if nargin < 3
        opt = [];
    end
    for i = 1:3
        if isfield(opt,optfields{i})
            if isempty(opt.(optfields{i}))
                opt.(optfields{i}) = defaults{i};
            end
        else
            opt.(optfields{i}) = defaults{i};
        end
    end
    x = x0(:);
    it = 0;
    F = feval(f, x);
    if ~(size(x,1) == size(F,1))
        error('f must return a column vector of the same size as x0')
    end
    normf = norm(F);
    J = jacobi(f,F,x);
    if nargout > 1
        ithist.x = [x(:)';zeros(opt.maxiter,length(x))];
        ithist.f = [F(:)';zeros(opt.maxiter,length(x))];
        ithist.normf = [normf;zeros(opt.maxiter,1)];
    end
    
    normdx = 2*opt.tolx;
    while(it < opt.maxiter+1 && normdx > opt.tolx && normf > opt.tolfun)
        if round(J) < 1e-15
            error('Singular jacobian at iteration %d\n',it)
        end
        dx = -J\F;
        normdx = norm(dx);
        if nargin > 3
            for j = 1:20
                j1 = find(x+dx<bounds(:,1));
                dx(j1) = dx(j1)/2;
                ju = find(x+dx>bounds(:,2));
                dx(ju) = dx(ju)/2;
                if isempty(j1) && isempty(ju)
                    break
                end
            end
        end
        x = x+dx;
        it = it+1;
        F = feval(f, x);
        normf = norm(F);
        J = J + F*dx'/(dx'*dx);
        if nargout > 1
            ithist.x(it+1,:) = x(:)';
            ithist.f(it+1,:) = F(:)';
            ithist.normf(it+1,:) = normf;
        end
    end
    I = (it)
    if it >= opt.maxiter && norm(F) > opt.tolfun
        warning('No convergence in %d iterations.\n',it+1)
    elseif normf>opt.tolfun
        warning('Newton step < %g, but function norm > %g\n',...
            opt.tolx,opt.tolfun)
    elseif normdx>opt.tolx
        warning('Funtion norm < %g, but newton step norm > %g\n',...
            opt.tolfun,opt.tolx)
    end
    if nargout > 1
        ithist.x(it+2:end,:) = [];
        ithist.f(it+2:end,:) = [];
        ithist.normf(it+2:end) = [];
    end
end
function J = jacobi(f,y0,x)
    delta = 1e-6*(max(1,sqrt(norm(x))));
    n = length(y0);
    m = length(x);
    J = zeros(n,m);
    for i = 1:m
        dx = zeros(m,1);
        dx(i) = delta/2;
        J(:,i) = (feval(f,x+dx)-feval(f,x-dx))/delta;
    end
end