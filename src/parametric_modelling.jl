module parametric_modelling

using linear_predictive_coding

export aryule, arcov, armcov
 
function aryule(x::Vector,L::Integer)
    if length(x) < L
        error("aryule: model order ($L) can't be more than length(x) ($(length(x))).")
    end
    R = xcorr(x,x)[length(x):]
    levinson([R,0],L)[1:2]
end

function arcov(x::Vector,L::Integer)
    if length(x) < 2*L
        error("arcov: model order ($L) can't be more than length(x)/2 ($(length(x))/2).")
    end
    N = length(x)
    phi = zeros(L+1,L+1)
    for k = 0:L, i=0:L, n=L:N-1
        phi[k+1,i+1] += x[n+1-k]*x[n+1-i]
    end
    phi = phi./(N-L)
 
    a = phi[2:end,2:end]\-phi[2:end,1]
    P = phi[1,1] + dot(vec(phi[1,2:end]),a)
 
    [1.,a],P
end
  
function armcov(x::Vector,L::Integer)
    if length(x) < 3/2*L
        error("armcov: length(x) ($(length(x))) must be greater than 3/2 the model order ((3/2)*$L).")
    end
    N = length(x)
    phi = zeros(L+1,L+1)
    for k = 0:L, i=0:L, n=0:N-L-1
        phi[k+1,i+1] += x[n+1+k]*x[n+1+i] + x[n+1+L-k]*x[n+1+L-i]
    end
    phi = phi./(2*(N-L))
 
    a = phi[2:end,2:end]\-phi[2:end,1]
    P = phi[1,1] + dot(vec(phi[1,2:end]),a)
 
    [1.,a],P
end

aryule(x::Matrix,L::Integer,dim::Integer=1) = ar(aryule,x,L,dim)
arcov(x::Matrix,L::Integer,dim::Integer=1) = ar(arcov,x,L,dim)
armcov(x::Matrix,L::Integer,dim::Integer=1) = ar(armcov,x,L,dim)

# this allows the ar* functions to handle matrices by applying
# the function 'method' along the specified dimension
function ar(method,x::Matrix,L::Integer,dim::Integer)
    N,M = size(x)
    if N==1 || M ==1 
        return method(vec(x),L)
    end
    if dim==1
        arparams = zeros(L+1,M)
        P = zeros(M)
        for i = 1:M
            arparams[:,i],P[i] = method(vec(x[:,i]),L)
        end
        return arparams,P
    else
        arparams = zeros(N,L+1)
        P = zeros(N)
        for i = 1:N
            arparams[i,:],P[i] = method(vec(x[i,:]),L)
        end
        return arparams,P
    end
end
 
x =[1,2,3,4,4,3,2,1]
for i = 4:10
print(i);print('\n')
a,P = aryule(x,i);
print(a);print('\n')
print(P);print('\n')
end
print(a)
print(P)

end # end module definition
