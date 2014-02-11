module linear_predictive_coding

import parametric_modelling: levinson, aryule

export lpc,rc,rc2lar,lar2rc,rc2is,is2rc,ac2poly,ac2rc,rc2poly,rc2ac,poly2rc,poly2ac

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

#ref: s.kay, modern spectral estimation, 1988, Prentice Hall, p170-173
function rc2poly(rc,R0=1)
    a = diagm(rc)
    L = length(rc)
    Gp = R0*prod(1-abs2(rc))  

    for m = 2:L
        a[m,1:m-1] = a[m-1,1:m-1]+a[m,m]*conj(a[m-1,m-1:-1:1])
    end
    [1 a[L,:]], Gp
end

#ref: s.kay, modern spectral estimation, 1988, Prentice Hall, p170-173
function rc2ac(rc,R0)
    a = diagm(rc)
    L = length(rc)
    P = zeros(L)
    P[1] = R0*(1-abs2(a[1,1]))
    # build up matrix of predictor coefficents
    for m = 2:L
        a[m,1:m-1] = a[m-1,1:m-1]+a[m,m]*conj(a[m-1,m-1:-1:1])
    end
    for k = 2:L
        P[k] = P[k-1]*(1-abs2(a[k,k]))
    end
    R = zeros(L) # AC coefficients, not including R0
    R[1] = -a[1,1]*R0
    for k = 2:L
        s = 0
        for l = 1:k-1
            s+= a[k-1,l]*R[k-l]
        end
        R[k] = -s - a[k,k]*P[k-1]
    end
    
    [R0, R]
end

#ref: s.kay, modern spectral estimation, 1988, Prentice Hall, p170-173
function poly2rc(poly::Vector,efinal=0)
    if poly[1] != 1
        poly ./= poly[1]
    end
    L = length(poly)-1
    a = zeros(L,L)
    P = zeros(L)
    P[L] = efinal
    a[L,:] = poly[2:end]
    for k = L:-1:2 
        for i = 1:k-1
            a[k-1,i] = (a[k,i] - a[k,k]*conj(a[k,k-i]))/(1-abs2(a[k,k]))
        end
        P[k-1] = P[k]/(1-abs2(a[k,k]))
    end
    R0 = P[1]/(1-abs2(a[1,1]))
    diag(a),R0
end

#ref: s.kay, modern spectral estimation, 1988, Prentice Hall, p170-173
function poly2ac(poly::Vector,efinal=0)
    if poly[1] != 1
        poly ./= poly[1]
    end
    L = length(poly)-1
    a = zeros(L,L)
    P = zeros(L)
    P[L] = efinal
    a[L,:] = poly[2:end]
    for k = L:-1:2 
        for i = 1:k-1
            a[k-1,i] = (a[k,i] - a[k,k]*conj(a[k,k-i]))/(1-abs2(a[k,k]))
        end
        P[k-1] = P[k]/(1-abs2(a[k,k]))
    end
    R = zeros(L) # AC coefficients, not including R0
    R0 = P[1]/(1-abs2(a[1,1]))
    R[1] = -a[1,1]*R0
    for k = 2:L
        s = 0
        for l = 1:k-1
            s+= a[k-1,l]*R[k-l]
        end
        R[k] = -s - a[k,k]*P[k-1]
    end
    
    [R0, R]
end


end # end module definition
