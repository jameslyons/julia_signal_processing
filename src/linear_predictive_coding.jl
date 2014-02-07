module linear_predictive_coding

import parametric_modelling: aryule

export levinson,lpc,rc,rc2lar,lar2rc,rc2is,is2rc

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

# lpc: compute linear prediction coefficients from signal 'x'
# returns a 2-tuple of linear prediction coefficients, Gp^2 of the filter
function lpc(x::Vector,L::Integer)
    aryule(x,L)
end

function lpc(x::Matrix,L::Integer,dim::Integer)
    aryule(x,L,dim)
end


function rc(x::Vector,L::Integer)
   if length(x) < L
        error("rc: model order ($L) can't be more than length(x) ($(length(x))).")
    end
    R = xcorr(x,x)[length(x):]
    levinson([R,0],L)[3]
end

function rc(x::Matrix,L::Integer,dim::Integer)
    N,M = size(x)
    if N==1 || M ==1 
        return rc(vec(x),L)
    end
    if dim==1
        refcoeff = zeros(L,M)
        for i = 1:M
            refcoeff[:,i] = rc(vec(x[:,i]),L)
        end
        return refcoeff
    else
        refcoeff = zeros(N,L)
        for i = 1:N
            refcoeff[i,:] = rc(vec(x[i,:]),L)
        end
        return refcoeff
    end
end

rc2lar(r::Number) = log((1+r)/(1-r))
@vectorize_1arg Number rc2lar

lar2rc(r::Number) = (exp(r)-1)/(exp(r)+1)
@vectorize_1arg Number lar2rc

rc2is(r::Number) = (2/pi)*asin(r)
@vectorize_1arg Number rc2is

is2rc(r::Number) = sin(pi*r/2)
@vectorize_1arg Number is2rc

function ac2poly(R::Vector)
    L = length(R)-1
    levinson(R,L)[1]
end

function ac2rc(R::Vector)
    L = length(R)-1
    levinson(R,L)[3]
end    




end # end module definition
