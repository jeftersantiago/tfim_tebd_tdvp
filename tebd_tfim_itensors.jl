let
using ITensors, ITensorMPS
using CSV
using DataFrames
using LinearAlgebra

    N = 50
    sites = siteinds("S=1/2", N) # , conserve_qns = true)
    J = 1.0

    hx = 0.9045
    hz = 0.809

    J = 4*J
    hx = 2*hx
    hz = 2*hz

    dt = 0.1
    tau = 20.0

    os = OpSum()

    for j = 1:N-1
        os += J, "Sz", j, "Sz", j+1
    end
    for j = 1:N
        os += hx,"Sx", j
        os += hz,"Sz", j
    end

    # Making gates
    odd_bond_gates = ITensor[]
    for j in 1:2:(N-1)
        hj = J * op("Sz", sites[j]) * op("Sz", sites[j+1])
        push!(odd_bond_gates, exp(-im * dt * hj))
    end
    even_bond_gates = ITensor[]
    for j in 2:2:(N-1)
        hj = J * op("Sz", sites[j]) * op("Sz", sites[j+1])
        push!(even_bond_gates, exp(-im * dt * hj))
    end
    field_gates_half = ITensor[]
    for j in 1:N
        hj = hx * op("Sx", sites[j]) + hz * op("Sz", sites[j])
        push!(field_gates_half, exp(-im * dt/2 * hj))
    end
    gates = [field_gates_half..., odd_bond_gates..., even_bond_gates..., field_gates_half...]

    cutoff = 1E-16
    psi = MPS(sites, "Dn")
    tlist = []
    O_expect = []
    for t in 0.0:dt:tau
        @show t
        Sz = expect(psi, "Sz")

        @time psi = apply(gates, psi; cutoff)
        normalize!(psi)

        push!(O_expect, 2 * sum(Sz)/N)
        push!(tlist, t)
    end
    df = DataFrame(time = collect(tlist), O = real.(O_expect))
    fname = "TEBD_obs_time_N=$(N)_tau=$(tau)_dt=$(dt)_J=$(J)_hx=$(hx)_hz=$(hz).csv"
    CSV.write(fname,  df)
end
