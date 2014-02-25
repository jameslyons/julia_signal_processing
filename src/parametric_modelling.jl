module parametric_modelling

export levinson, aryule, arcov, armcov

# levinson: levinson durbin recursion
# takes vector of autocorrelation coefficients 'R' and
# returns a 3-tuple of linear prediction coefficients, Gp^2 of the filter, 
# and reflection coefficents
function levinson(R::Vector,L::Integer)
    a = zeros(L,L)
    P = zeros(1,L)
 
    # for m = 1
    a[1,1] = -R[2]/R[1]
    P[1] = R[1]*(1-a[1,1]^2)
 
    # for m = 2,3,4,..L
    for m = 2:L
        a[m,m] = (-(R[m+1] + dot(vec(a[m-1,1:m-1]),R[m:-1:2]))/P[m-1])
        a[m,1:m-1] = a[m-1,1:m-1]+a[m,m]*a[m-1,m-1:-1:1]
        P[m] = P[m-1]*(1-a[m,m]^2)
    end
    [1., vec(a[L,:])], P[L], diag(a)
end


function aryule(x::Vector,L::Integer)
    if length(x) < L
        error("aryule: model order ($L) can't be more than length(x) ($(length(x))).")
    end
    R = xcorr(x,x)[length(x):]
    levinson([R,0],L)[1:2]
end

function arburg(x::Vector,p::Integer)
    N = length(x)

    P = zeros(1,p+1)
    f = zeros(p+1,N)
    b = zeros(p+1,N)
    a = zeros(p,p)

    # Initialisation
    f[1,:] = x
    b[1,:] = x
    P[1] = var(x)

    for m = 1:p
        numer,denom = 0,0
        for n=0:N-m-1
            numer += b[m,n+1]*f[m,n+2]
            denom += f[m,n+2]^2 + b[m,n+1]^2
        end
        a[m,m] = -2*numer/denom
        for i = 1:m-1
            a[m,i] = a[m-1,i] + a[m,m]*a[m-1,m-i]
        end
        P[m+1] = P[m]*(1-a[m,m]^2)
        for n=0:N-m-1
            f[m+1,n+1] = f[m,n+2] + a[m,m]*b[m,n+1]
            b[m+1,n+1] = b[m,n+1] + a[m,m]*f[m,n+2]
        end
    end
    [1 a[p,:]], P[p]
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
arburg(x::Matrix,L::Integer,dim::Integer=1) = ar(arburg,x,L,dim)

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
