let
    using QuantumToolbox
    using LinearAlgebra
    using DataFrames
    using CSV

    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    N = 10
    J = 1.0
    hx = 0.9045
    hz = 0.809

    H_zz = sum(1:N-1) do i
        J * multisite_operator(Val(N), i => sz, i+1 => sz)
    end
    H_x = sum(1:N) do i
        hx * multisite_operator(Val(N), i => sx)
    end
    H_z = sum(1:N) do i
        hz * multisite_operator(Val(N), i => sz)
    end

    O = (1/N) * sum(1:N) do i
        multisite_operator(Val(N), i => sz)
    end

    H_tfim = H_zz + H_x + H_z
    # @show minimum(eigvals(H_tfim))

    psi_0 = kron(fill(basis(2, 1), N)...)

    tau = 20.0
    tlist = range(0, tau, length = 100)

    result = sesolve(H_tfim, psi_0, tlist, progress_bar=Val(true))
    psi_t = result.states
    O_expect = expect.(Ref(O), psi_t)

    df = DataFrame(
        time = collect(tlist),
        O = real.(O_expect))

    fname = "ED_obs_time_N=$(N)_tau=$(tau)_J=$(J)_hx=$(hx)_hz=$(hz).csv"
    CSV.write(fname,  df)
end
