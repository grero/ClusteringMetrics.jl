using ClusteringMetrics
using Test

Z1 = fill(0.0, 2, 10)
Z1[:,1:5] = [0.1 0.2 0.15 0.2 0.22;
             0.15 0.21 0.13 0.21 0.19]
Z1[:,6:end] = Z1[:,1:5] .- [1.0;1.0]
ϵ = [-0.010114902986680585 0.004463792650270601 -0.00153613750975559 -0.0055050913393961485 0.0035611946194614437 -0.0009473083769016829 -0.0011402700190551592 0.00043139456531041 0.007363905500805968 -0.0036852874409562726;
     -0.004970181279034265 -0.008439692559205151 -0.002063148403333957 0.0027397663153164308 -0.009655682877822354 0.002952749805508587 0.010722686475465308 -0.008101446035381798 0.0024392528878517524 -0.007034803358359168]
Z1 .+= ϵ
label1 = [fill(1,5);fill(2,5)]

@testset "Isolation distance" begin
    # large isolation distance
    d = ClusteringMetrics.get_isolation_distance(Z1, label1)
    @test d ≈  [7346.57515900546, 11695.14306811699]
end

@testset "Cluster matching" begin
    Z2 = fill!(similar(Z1), 0.0)
    Z2 .= Z1
    label2 = label1
    dd = ClusteringMetrics.get_isolation_distance(Z1, label1, Z2, label2)
    # large cross distance, indicating high generalisability
    @test dd ≈ [296.9888641641518, 301.87857155988695]
    Z2[:,1:5] = Z1[:,6:end]
    Z2[:, 6:end] = Z1[:, 1:5]
    dd = ClusteringMetrics.get_isolation_distance(Z1, label1, Z2, label2)
    # low cross distance indicating low generlisability
    @test dd ≈ [0.0033671296154972326, 0.003312590207488839]
end
