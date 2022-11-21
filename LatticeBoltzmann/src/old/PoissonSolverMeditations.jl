using FFTW
using Statistics: mean
using Plots
using FiniteDifferences: central_fdm
using Interpolations
using ForwardDiff: gradient
using DiffEqOperators
## ΔΦ = αρ
function λ1(i, n, L)
    if i == 1
        return 0
    end
    -(2*π*fftfreq(n)[i]/L*n)^(-2)
end

#λ1(n) = λ1.(1:n-1,n)
λ(n, L=1) = λ1.(1:n,n, L)
#plot(λ1(200))

# ΔΦ = αρ
function solve_f(rho, L, alpha)
    return alpha*real.(ifft(fft(rho) .* λ(length(rho),L) ))
    #return ifft(ifftshift(fftshift(fft(rho)) .* λ1(length(rho))))
end

function num_diff(y, degree, approx_order, dx)
    D = CenteredDifference(degree, approx_order, dx, length(y))
    q = PeriodicBC(typeof(y))
    D*(q*y)
end

function gaussian(x, μ=0,σ=1, A=1)
    A * exp(-((x - μ) / σ)^2)
end
##
n = 256
X_min = -2*π
X_max = -X_min
L = X_max - X_min
alpha = 1
x = LinRange(X_min,X_max,n+1)[1:end-1]
#rho = sin.(rho_0) ./ alpha
#rho = 12*((rho_0.-π).^2)/alpha
μ = 0
σ = 1
A = 1
phi_0 = gaussian.(x,μ,σ,A) #Test gaussiano
#phi_0 =  gaussian.(x,-1.5,0.5) + gaussian.(x,1.5,0.5) + gaussian.(x,0,1.1,0.5)
#phi_0 = 4 .*sin.(5 .*x) + 3 .* cos.(3 .*x)
phi_0 = phi_0 .- mean(phi_0)
#plot(x,phi_0)
#
diff_order = 2
approx_order = 11
dx = x[2]-x[1]

using DiffEqOperators: PeriodicBC
q = PeriodicBC(typeof(phi_0))

rho = num_diff(phi_0,diff_order,approx_order,dx)
#phi = solve_f(y1, L, alpha, n)
#plot(rho_0,[phi, phi_0])
#
phi_final = solve_f(rho .-mean(rho), L, alpha) .+ mean(rho)
plot(x,[phi_final, phi_0], label = ["numerical" "analytical"])
#sum(abs.(phi_final-phi_0))


##


function /̃(x::Number, y::Number)
    if x == one(typeof(x)) && y == zero(typeof(y))
        zero(typeof(y))
    else
        Base.:(/)(x, y)
    end
end

function λ2(i, n, L)
    if i == 1
        return 0
    end
    -(2*π*fftfreq(n)[i]/L*n)^(2)
end

λ(n, m, L=1) = λ2.(1:n,n, L) .+ λ2.(1:n,n, L)'
λ(n, m, l,  L=1) = (λ2.(1:n,n, L) .+ λ2.(1:n,n, L)') .+ λ2.(1:n,n, L)'

1.0 ./̃λ(20,20)
1.0 ./̃λ(20,20,20)
##

function solve_f(rho, L, alpha)
    return alpha*real.(ifft(fft(rho) ./ λ(length(rho),L) ))
    #return ifft(ifftshift(fftshift(fft(rho)) .* λ1(length(rho))))
end

function num_diff(y, degree, approx_order, dx)
    D = CenteredDifference(degree, approx_order, dx, length(y))
    q = PeriodicBC(typeof(y))
    D*(q*y)
end

function gaussian(x, μ=0,σ=1, A=1)
    A * exp(-((x - μ) / σ)^2)
end
##
#λ1(n) = λ1.(1:n-1,n)
λ1(n, L=1) = λ1.(1:n,n, L)


#λ2.(1:n, 1:n, n, L)

xy = x*transpose(x)


f(x,y) = gaussian(x,0,0.1) * gaussian(y,1,2)


plot(x,x,f.(x',x))
plot(x,x,f,st=:contour)
