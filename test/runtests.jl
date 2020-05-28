using ClusteringMetrics
using Test


@testset "Cluster matching" begin
    Z1 = fill(0.0, 2, 10)
    Z1[:,1:5] = [0.1 0.2 0.15 0.2 0.22;
                 0.15 0.21 0.13 0.21 0.19]
    Z1[:,6:end] = Z1[:,1:5] .- [1.0;1.0]
    label1 = [fill(1,5);fill(2,5)]
    Z2 = fill!(similar(Z1), 0.0)
    Z2 .= Z1
    label2 = label1
    dd = ClusteringMetrics.get_isolation_distance(Z1, label1, Z2, label2)
    @test dd ≈ [338.83783783783787, 338.83783783783826]
    Z2[:,1:5] = Z1[:,6:end]
    Z2[:, 6:end] = Z1[:, 1:5]
    dd = ClusteringMetrics.get_isolation_distance(Z1, label1, Z2, label2)
    @test dd ≈ [0.002951264257796922, 0.0029512642577969195]
end
