## Δρ = αρ
function λ1(i, n, L)
    if i == 1
        return 0
    end
    -(2*π*fftfreq(n)[i]/L*n)^(-2)
end

λ(n, L=1) = λ1.(1:n,n, L)

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
