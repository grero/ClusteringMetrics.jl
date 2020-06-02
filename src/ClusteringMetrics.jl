module ClusteringMetrics
using StatsBase
using MultivariateStats
using LinearAlgebra

"""
    get_cluster_distance(Z1::Matrix{T}, label1::Vector{T2}, Z2::Matrix{T}, label2::Vector{T2}) where T <: Real where T2

Compute the distance between the class clusters in `Z1` and `Z2` using class labels `label1` and `label2`, normalized by the the total intra-class variance. 
"""
function get_cluster_distance(Z1::Matrix{T}, label1::Vector{T2}, Z2::Matrix{T}, label2::Vector{T2}) where T <: Real where T2
    # Compute class statistics on the projected trials
    nc1 = maximum(label1)
    nc2 = maximum(label2)
    ldastats_1 = MultivariateStats.multiclass_lda_stats(nc1, Z1, label1)
    ldastats_2 = MultivariateStats.multiclass_lda_stats(nc2, Z2, label2)

    # Compute a normalized distance between cluster centers
    Sw = Diagonal(diag(ldastats_1.Sw + ldastats_2.Sw))
    d = ldastats_1.cmeans - ldastats_2.cmeans
    D = diag(d'*inv(Sw)*d)
end

"""
    get_cluster_confusion(Z1::Matrix{T}, label1::Vector{T2}, Z2::Matrix{T}, label2::Vector{T2}) where T <: Real where T2

Compute the distance to the nearest cluster normalied to the distance to the true cluster.    
"""
function get_cluster_confusion(Z1::Matrix{T}, label1::Vector{T2}, Z2::Matrix{T}, label2::Vector{T2}) where T <: Real where T2

    # Compute class statistics on the projected trials
    ldastats_1 = CodeMorphingCore.MultivariateStats.multiclass_lda_stats(4, Z1, label1)
    ldastats_2 = CodeMorphingCore.MultivariateStats.multiclass_lda_stats(4, Z2, label2)
    nc1= ldastats_1.nclasses
    nc2= ldastats_2.nclasses
    d = fill(0.0,nc1, nc2)
    for j in 1:nc2
        for i in 1:nc1
            d[i,j] = ldastats_1.cmeans[:,i] - ldastats_2.cmeans[:,j]
        end
    end

    d_true = diag(d) 
    d_nearest = minimum(d, dims=2)
    d_nearest./d_true
end

"""
    get_isolation_distance(Z::Matrix{T}, label::Vector{T2}) where T <: Real where T2
        
For each point in `Z` compute the distance to its nearest neightbour in its own class and compare that to the its nearest neighbour outside its class.
"""
function get_isolation_distance(Z::Matrix{T}, label::Vector{T2}) where T <: Real where T2
    nc = maximum(label)
    d_in = fill(Inf, nc)
    d_out = fill(Inf, nc)
    np = size(Z,2)
    for i in 1:np 
        li = label[i]
        dii = d_in[li]
        doi = d_out[li]
        for j in 1:np
            if i == j
                continue
            end
            lj = label[j]
            d = sum(x->x*x, Z[:,i] - Z[:,j])
            if li == lj
                dii = min(dii, d)
            else
                doi = min(doi, d)
            end
        end
        d_in[li] = dii
        d_out[li] = doi
    end
    d_out./d_in
end

"""
    et_isolation_distance(Z1::Matrix{T}, label1::Vector{T2}, Z2::Matrix{T}, label2::Vector{T2}) where T <: Real where T2

Compute the stimilarly of two clusterings by comparing the nearest point between identical clusters to nearest points between clusters. 

A large value means that points for that class in one set are closer to the points of the same class in the other, compared to points in other classes. Intuitively, a large value means that the clustering generalizes across the two sets, while a small value means that the classes are mixed between the two sets.
"""
function get_isolation_distance(Z1::Matrix{T}, label1::Vector{T2}, Z2::Matrix{T}, label2::Vector{T2}) where T <: Real where T2
    nc = maximum(label1)
    nc == maximum(label2) || error("Cluster labels should match between the two inputs")

    np1 = size(Z1, 2)
    np2 = size(Z2, 2)

    n1 = StatsBase.countmap(label1)
    n2 = StatsBase.countmap(label1)
    d_in = fill(0.0, nc) 
    d_out = fill(Inf,np1*np2,nc)
    k = 1
    for i in 1:np1
        li = label1[i]
        for j in 1:np2
            lj = label2[j]
            d = sum(x->x*x, Z1[:,i] - Z2[:,j])
            if li == lj
                d_in[li] += d # add to the mean for this class
            else
                d_out[k,li] = d
            end
            k += 1
        end
    end
    d_in ./= [n1[k]*n2[k] for k in 1:nc]
    sort!(d_out, dims=1)
    d_om = [mean(d_out[1:n1[k]*n2[k],k]) for k in 1:nc]
    d_om./d_in
end

"""
    get_cluster_isolation(Z1::Matrix{T}, label1::Vector{T2}, Z2::Matrix{T}, label2::Vector{T2}) where T <: Real where T2

For each cluster, return ratio of the number of points `Z1` that are closer to a point in the same cluster in `Z2` to the number of points that are closer to a different cluster in `Z2`.
"""
function get_cluster_isolation(Z1::Matrix{T}, label1::Vector{T2}, Z2::Matrix{T}, label2::Vector{T2}) where T <: Real where T2
    nc = maximum(label1)
    nc == maximum(label2) || error("Cluster labels should match between the two inputs")

    np1 = size(Z1, 2)
    np2 = size(Z2, 2)

    n1 = StatsBase.countmap(label1)
    n2 = StatsBase.countmap(label2)
    n_in = fill(0, nc)
    n_out = fill(0,nc)
    for i in 1:np1
        li = label1[i]
        d_in = Inf
        d_out = Inf
        for j in 1:np2
            lj = label2[j]
            d = sum(x->x*x, Z1[:,i] - Z2[:,j])
            if lj == li
                d_in = min(d_in, d)
            else
                d_out = min(d_out,d)
            end
        end
        if d_in < d_out
            n_in[li] += 1
        else
            n_out[li] += 1
        end
    end
    q = (n_in - n_out)./(n_in + n_out)
    #normalize from 0 to 1
    q = (q .+ 1)./2
end

end # module
