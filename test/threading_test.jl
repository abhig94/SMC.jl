# threading vs distributed parallelism

using ModelConstructors, SMC
using LinearAlgebra, PDMats, Distributions
using Printf, Distributed, Random, HDF5, FileIO, JLD2

cd("C:\\Users\\Abhi\\.julia\\dev\\SMC\\test")
include("C:\\Users\\Abhi\\.julia\\dev\\SMC\\test\\modelsetup.jl")

Random.seed!(42)

## set up model and basics
m = setup_linear_model()
parameters = m.parameters
n_parts = m.settings[:n_particles].value
# loglik_fn is the loglikelihood function

# Read in generated data
data = h5read("reference/test_data.h5", "data")

# set up cloud
init_cloud = SMC.Cloud(length(m.parameters), get_setting(m,:n_particles))
one_draw_closure() = SMC.one_draw(loglik_fn, parameters, data)

# initial draw
Random.seed!(42)
draws, loglh, logprior = SMC.vector_reduce([one_draw_closure() for i in 1:n_parts]...)
SMC.initial_draw!(loglik_fn, parameters, data, init_cloud; parallel = false)




## test initial_draw!
Random.seed!(42)
test_c1 = SMC.Cloud(length(m.parameters), 100)
SMC.initial_draw!(loglik_fn, parameters, data, test_c1; parallel = false)

Random.seed!(42)
test_c2 = SMC.Cloud(length(m.parameters), 100)
SMC.initial_draw!(loglik_fn, parameters, data, test_c2; parallel = true)

SMC.get_vals(test_c1) ≈ SMC.get_vals(test_c2)

## test drawing
# distributed
@everywhere Random.seed!(42)
draws_d, loglh_d, logprior_d =  @sync @distributed (SMC.vector_reduce) for i in 1:n_parts
        one_draw_closure()
    end

# threaded (still bad?)
draw_1, loglh_1, logprior_1 = one_draw_closure()
@everywhere Random.seed!(42)
draws_t, loglh_t, logprior_t = repeat(draw_1, 1, n_parts), repeat(loglh_1, 1, n_parts), repeat(logprior_1, 1, n_parts) # preallocate
Threads.@threads for i in 1:n_parts
        draws_t[:,i], loglh_t[:,i], logprior_t[:,i] = one_draw_closure()
    end

# not safe
@everywhere Random.seed!(42)
out = []
Threads.@threads for i in 1:n_parts
    push!(out,one_draw_closure())
end
size(out)[1] == n_parts # no guarantee that this is true
draws_t2, loglh_t2, logprior_t2 = SMC.vector_reduce(out...)


# check that σ draws are valid
@assert all(draws[[3,6,9],:] .> 0)
@assert all(draws_d[[3,6,9],:] .> 0)
@assert all(draws_t[[3,6,9],:] .> 0)


## test initialize_likelihoods!
# outside of cloud
draw_likelihood_closure(draw::Vector{Float64}) = SMC.draw_likelihood(loglik_fn, copy(parameters), data, draw)

loglh, logprior = SMC.scalar_reduce([draw_likelihood_closure(draws[:, i]) for i in 1:n_parts]...)

loglh_t, logprior_t = zeros(n_parts), zeros(n_parts) # preallocate
x = zeros(n_parts)
Threads.@threads for i in 1:n_parts
    loglh_t[i,:], logprior_t[i,:] = draw_likelihood_closure(draws[:,i])
    x[i] = i
end

count(.!(loglh .≈ loglh_t)) == 0
count(.!(logprior .≈ logprior_t)) == 0
count(.!(1:n_parts .== x)) == 0

# testing actual functions
cloud_1 = deepcopy(init_cloud)
cloud_2 = deepcopy(init_cloud)

SMC.initialize_likelihoods!(loglik_fn, parameters, data, cloud_1)
SMC.initialize_likelihoods!(loglik_fn, parameters, data, cloud_2; parallel = true)  # parallel = false is default

@assert SMC.get_loglh(cloud_1) ≈ SMC.get_loglh(cloud_2)
@assert SMC.get_loglh(cloud_1) ≈ SMC.get_loglh(init_cloud)
@assert SMC.get_logprior(cloud_1) ≈ SMC.get_logprior(cloud_2)


## resample 
# Stores cumulative weights until given index

Random.seed!(42)
weights = rand(n_parts)

cumulative_weights = cumsum(weights ./ sum(weights))
offset = rand(n_parts)

indx = Vector{Int64}(undef, n_parts)
indx_d = similar(indx)
indx_t = similar(indx)

# default
for i in 1:n_parts
    indx[i] = findfirst(x -> offset[i] < x, cumulative_weights)
end
# distributed
indx = @sync @distributed (vcat) for i in 1:n_parts
    findfirst(x -> offset[i] < x, cumulative_weights)
end
# threaded
Threads.@threads for i in 1:n_parts
    indx_t[i] = findfirst(x -> offset[i] < x, cumulative_weights)
end

# test
indx_d == indx
indx_t == indx


# using actual function
Random.seed!(42)
indx_1 = SMC.resample(weights; method = :multinomial, parallel = false)

Random.seed!(42)
indx_2 = SMC.resample(weights; method = :multinomial, parallel = true)

indx_1 == indx_2


## Mutation 
θ_bar = SMC.weighted_mean(init_cloud)
R     = SMC.weighted_cov(init_cloud)

n_blocks = get_setting(m, :n_smc_blocks)
n_free_para = length(m.parameters)
blocks_free = SMC.generate_free_blocks(n_free_para, n_blocks)
blocks_all  = SMC.generate_all_blocks(blocks_free, collect(1:n_free_para))

ϕ_n = 0.0001
ϕ_n1 = 0.0
c = 0.5
α = 1.0


new_particles = similar(init_cloud.particles')
new_particles_t = similar(init_cloud.particles')

function mutation_closure(k::Int64)
    Random.seed!(42)
    SMC.mutation(loglik_fn, deepcopy(m.parameters), data, init_cloud.particles[k, :], θ_bar, R, n_free_para,
                 blocks_free, blocks_all, ϕ_n, ϕ_n1; c = c, α = α,
                 n_mh_steps = 1)
end

Threads.@threads for k in 1:n_parts
    new_particles_t[:,k] = mutation_closure(k)
end

new_particles = hcat([mutation_closure(k) for k=1:n_parts]...)

new_particles ≈ new_particles_t

