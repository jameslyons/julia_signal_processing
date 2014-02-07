module linear_predictive_coding

export levinson

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
function lpc(x,L)
   if length(x) < L
        error("lpc: model order ($L) can't be more than length(x) ($(length(x))).")
    end
    R = xcorr(x,x)[length(x):]
    levinson([R,0],L)[1:2]
end

function rc(x,L)
   if length(x) < L
        error("rc: model order ($L) can't be more than length(x) ($(length(x))).")
    end
    R = xcorr(x,x)[length(x):]
    levinson([R,0],L)[3]
end

function rc2lar(r::Number)
    return log((1+r)/(1-r))
end

@vectorize_1arg Number rc2lar

function lar2rc(r::Number)
    return (exp(r)-1)/(exp(r)+1)
end

@vectorize_1arg Number lar2rc







end # end module definition
