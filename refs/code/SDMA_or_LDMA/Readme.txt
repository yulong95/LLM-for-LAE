This simulation code package is mainly used to reproduce the results of the following paper [1]:

[1] Z. Wu and L. Dai, "Multiple access for near-field communications: SDMA or LDMA?," IEEE J. Sel. Areas Commun., vol. 41, no. 6, pp. 1918-1935, Jun. 2023.

The conference version of this journal paper is [2]:

[2] Z. Wu and L. Dai, "Location Division Multiple Access for Near-Field Communications," IEEE Int. Conf. Commun. (IEEE ICC'23), Rome, Italy, May 2023.
*********************************************************************************************************************************
If you use this simulation code package in any way, please cite the original paper [1] above. 

The author in charge of this simulation code pacakge is: Zidong Wu (email: wuzd19@mails.tsinghua.edu.cn).

Reference: We highly respect reproducible research, so we try to provide the simulation codes for our published papers (more information can be found at: 
http://oa.ee.tsinghua.edu.cn/dailinglong/publications/publications.html). 

Please note that the MATLAB R2019b is used for this simulation code package,  and there may be some imcompatibility problems among different software versions. 

Copyright reserved by the Broadband Communications and Signal Processing Laboratory (led by Dr. Linglong Dai), Tsinghua National Laboratory
for Information Science and Technology (TNList), Department of Electronic Engineering, Tsinghua University, Beijing 100084, China. 

*********************************************************************************************************************************
Abstract of the paper: 

Spatial division multiple access (SDMA) is essential to improve the spectrum efficiency for multi-user multiple-input multiple-output (MIMO) communications. The classical SDMA for massive MIMO with hybrid precoding heavily relies on the angular orthogonality in the far field to distinguish multiple users at different angles, which fails to fully exploit spatial resources in the distance domain. With the dramatically increasing number of antennas, the extremely large-scale antenna array (ELAA) introduces additional resolution in the distance domain in the near field. In this paper, we propose the concept of location division multiple access (LDMA) to provide a new possibility to enhance spectrum efficiency compared with classical SDMA. The key idea is to exploit extra spatial resources in the distance domain to serve different users at different locations (determined by angles and distances) in the near field. Specifically, the asymptotic orthogonality of near-field beam focusing vectors in the distance domain is proved, which reveals that near-field beam focusing is able to focus signals on specific locations with limited leakage energy at other locations. This special property could be leveraged in hybrid precoding to mitigate inter-user interferences for spectrum efficiency enhancement. Moreover, we provide the spherical-domain codebook design method for LDMA communications with the uniform planar array, which provides the sampling method in the distance domain. Additionally, performance analysis of LDMA is provided to reveal that the asymptotic optimal spectrum efficiency could be achieved with the increasing number of antennas. Finally, simulation results verify the superiority of the proposed LDMA over SDMA in different scenarios.

*********************************************************************************************************************************
How to use this simulation code package?

1. "Fig4_Plot_correlation": Run this file to obtain the results in Fig. 4.
2. "Fig9_Compare_LDMA_linear_distribution": ULA and linear distributed users. Run this file to obtain the results in Fig. 9.
3. "Fig10_Compare_LDMA_uniform_distribution": ULA and uniform distributed users. Run this file to obtain the results in Fig. 10.
4. "Fig11_Compare_LDMA_uniform_L": ULA and uniform distributed users with different number of NLoS paths. Run this file to obtain the results in Fig. 11.
5. "Fig12_Compare_LDMA_uniform_kappa": ULA and uniform distributed users with different Rician ratios. Run this file to obtain the results in Fig. 12.
6. "Fig13_Compare_LDMA_uniform_N": ULA and uniform distributed users with different number of antennas. Run this file to obtain the results in Fig. 13.
7. "Fig14_generate_UPA_Codebook": run this file before running the UPA simulation codes, which generates the required codebook according to different delta.
8. "Fig14a_Compare_LDMA_UPA_linear_distribution": UPA and linear distributed users. Run this file to obtain the results in Fig. 14(a).
9. "Fig14b_Compare_LDMA_UPA_uniform_distribution": UPA and uniform distributed users. Run this file to obtain the results in Fig. 14(b).
10. "Fig14_generate_uniform_codebook": run this file before running the UPA uniform codebook codes in the follow-up.
11. "Fig14a_Compare_LDMA_UPA_uniform_codebook_linear": UPA with uniform codebook and linear distributed users. Run this file to obtain the results in Fig. 14(a).
12. "Fig14b_Compare_LDMA_UPA_uniform_codebook": UPA with uniform codebook and uniform distributed users. Run this file to obtain the results in Fig. 14(b).

Basic parameters for Fig.14
1) UPA: 256 * 16 at 30 GHz
2) Number of users: 4
3) Spherical-domain codebook: delta is set to 0.55 to generate the codebook. 
4) Channel: 5 paths including one LoS path and 4 NLoS paths. Rician factor is set to 8.
5) Linear distribution: theta in [pi/3, 2pi/3], phi in [-pi/6, pi/6], r in [4m, 50m] and users are linearly distributed in the same direction.
6) Uniform distribution: theta in [pi/3, 2pi/3], phi in [-pi/6, pi/6], r in [4m, 50m] and users are uniformly distributed in the 3D space.

*********************************************************************************************************************************
Enjoy the reproducible research!


