# SaDLIO — novelty and contribution survey

Written: 2026-10-04 · Status: literature survey, no code · Companion to
[`VIO_fix_+_LIO_Prospects.md`](VIO_fix_+_LIO_Prospects.md) (Part 4)

This document asks one question: what could make a LiDAR-inertial "Sparsify and Densify" paper
(SaDLIO) publishable, given what got SaDVIO published and what already exists for LiDAR.

**Method and confidence.** Five parallel literature searches, one per topic:

1. LIO back ends and sparsification;
2. LiDAR densification and meshing;
3. planetary traversability and datasets;
4. LiDAR reflectivity as a photometric signal;
5. estimators whose landmarks are also the map.

Together they ran about 200 web searches (the session budget, reached partway through), plus
arXiv-API queries and abstract or full-text reads. IEEE Xplore full text, Scopus and
Chinese-language venues were not searched systematically. Every "not found" below therefore means
**medium confidence** that it is absent, not proof. Cited URLs were opened or seen in results by
the searching agents; this document has not re-opened every one. Read the closest prior art in
full before writing any novelty sentence (list in §6).

---

## 0. Context: why SaDLIO (MeshCSLAM)

*Added 2026-10-04 after reading `~/EXECO/DeTest-EXECO` (`MeshCSLAM_Discussion.md`,
`MeshVLPR_RAL_draft.md`, `place-recognition-next-steps.md`).*

The motivation is not a standalone LIO paper. It is to give the **drone and the rover in MeshCSLAM
the same architecture**. Today:

| | Drone | Rover |
|---|---|---|
| Odometry | SaDVIO (sliding window, IMU, marginalization) | KISS-ICP (no IMU, no window; 5.8 cm RMS over 173 m in the corridor) |
| Dense map | `MarginalDepthInjector` → FFS depth → TSDF, when a KF leaves the window | separate `submap_builder` TSDF on KISS-ICP poses |
| Output to the map server | submap meshes + poses | submap meshes + poses |

**What has to be symmetric.** MeshVLPR found that the place-recognition gain comes from **using the
same surface representation on both robots** (consistent normals), not from how each robot
estimates its pose. The map server (iSAM2 + GNC/PCM) needs, from each robot:

- KF/submap-anchored meshes that can be moved when poses are optimized;
- odometry factors with usable covariances;
- gravity (for traversability and the gravity-aligned place-recognition variant).

The symmetry that matters is therefore **the back end and the products**, not the front end.

**"SaDLIO" for MeshCSLAM = the SaDVIO back end with a LiDAR front end.**

- Same sliding window, IMU preintegration and marginalization.
- Same marginalization hook: when a KF leaves the window, its deskewed scan is fused at the KF's
  smoothed pose into the same TSDF/VDB-GPDF backend. The `lidar` preset already gives 1.3 cm /
  F 90.2 on ground-truth poses ([`dense_vdb_gpdf.md`](dense_vdb_gpdf.md)).
- Both robots then emit the same things: KF-anchored submap meshes, plus odometry factors from the
  same formulation.

Two build levels:

- **T1, loosely coupled (weeks).** A LiDAR scan-to-map front end (KISS-ICP style) produces
  relative-pose measurements, added as relative-pose factors in the SaDVIO window next to IMU
  preintegration. No LiDAR landmarks. Sparsification becomes pose-only (absolute + relative-pose
  topology). The LiDAR registration covariance has to be estimated (e.g. from the point-to-plane
  Hessian).
- **T2, tightly coupled plane landmarks (months).** This is R1 below. Only worth it if SaDLIO
  is meant to be a paper in its own right. Its plane-patch surface is also *not* the TSDF mesh that
  MeshVLPR was validated on.

**Where the research contribution moves.** For MeshCSLAM, SaDLIO is an enabling component and a
design choice, so it does not need novelty of its own. The S and D ideas can instead earn novelty
*inside* the C-SLAM paper:

- **S as the inter-robot exchange format (hypothesis, not yet searched).** Each robot already
  computes a KLD-sparsified summary of what it marginalizes. Sending sparse relative factors between
  submap anchors, with the recovered information, would give the server consistent odometry
  covariances from both heterogeneous estimators by one formulation.
  - Closest known prior art: Basalt NFR (single robot), SLIM (multi-session LiDAR map merging).
  - **Needs its own literature check against C-SLAM back ends** (Kimera-Multi, Swarm-SLAM, DOOR-SLAM,
    COVINS, LAMP 2.0) before any claim.
- **D as KF-anchored submap meshes that move with the optimized poses.** This is already step 6
  of the MeshCSLAM map server.

**Checks before building T1.**

- Run KISS-ICP on CPLite and Factory. If it already holds there, the gain from an IMU is gravity,
  robustness and symmetry, not accuracy; say that honestly.
- Check whether the AirSim VLP-16 gives per-point timestamps, and how it sweeps (deskewing).
- Finish freezing SaDVIO first: T1 reuses its IMU and window code.

---

## 1. What got SaDVIO published (summary)

From the paper (`RAL_2024___SaDVIO.pdf`):

- **Main contribution: Densify.** Delaunay on tracked features, a per-triangle photometric planarity
  test (ZNCC of a plane-warped patch, left to right), and ray casting on the triangle soup with the
  best-ZNCC triangle. This gives a dense cloud without dense stereo (39 ms vs 162 ms for SGM),
  beats Kimera's mesher, and has an ablation with and without ray casting.
- **Application framing.** CPU-only planetary rover, GESTALT traversability, and the MER heritage
  of expensive dense stereo.
- **New open dataset** (Mars yard at CNES, published on the dataverse as "NUANCES - Mars Yard
  dataset", DOI 10.34849/DGLZPE), plus open-source code.
- **Sparsify was incremental.** The paper itself says the VIO topology is "similar to" Hsiung et
  al. 2018 and that the topologies are *assumed* to approximate the dense prior well. The only
  evidence offered is runtime.
- Odometry accuracy was "good enough", not state of the art. The map was the contribution.

The recipe was: a real compute constraint, one technical idea that addresses it, an ablation,
comparison with the closest system, a dataset, and open-source code.

---

## 2. Already done: these cannot be novelty claims

| Idea | Prior art |
|---|---|
| Triangulating a LiDAR scan in polar/spherical coordinates for rover planning | Gingras et al. CRV 2010; Rekleitis et al. AuRo 2013 |
| Fast meshing on the range-image grid with edge-length / discontinuity rejection | Holz & Behnke RAS 2014 (PCL `OrganizedFastMesh`); Polylidar3D 2020; Guinard & Vallet 2018 |
| Meshing a single scan to upsample it | CURL, RSS 2022 (**strongest threat for "D"; read in full**) |
| Classical CPU range-image interpolation of 16-ring LiDAR, spurious points filtered, improves odometry | Velasco-Sánchez et al. ETFA 2023 (ROS package); You & Kim 2022; IP-Basic 2018 |
| Uncertainty-filtered LiDAR upsampling | Shan et al. 2020, TULIP CVPR 2024 (MC dropout); DU-OR 2025 |
| Real-time CPU LiDAR meshing and mesh-based registration | SLAMesh ICRA 2023; ImMesh T-RO 2023; CAD-Mesher; LGFaware; IDTMM |
| Per-triangle confidence for measured vs gap-filling triangles | Caraffa, Brédif, Vallet ACCV 2016 |
| Intensity-patch NCC with a range-based occlusion check (for odometry) | COIN-LIO ICRA 2024; PG-LIO 2025; RI-LIO 2023 |
| Intensity texture stored on local 3D surfaces | COIN-BIEVR (ICRA 2026 workshop) |
| KLD/NFR sparsification of a fixed-lag window prior | Hsiung et al. IROS 2018 (VIO); Debeunne et al. IROS 2023; SaDVIO |
| NFR with LiDAR plane/line landmarks | SLIM (T-RO 2025), long-term multi-session mapping |
| Chow-Liu approximate marginalization in laser SLAM | Kretzschmar & Stachniss IJRR 2012 (2D, long-term) |
| Fixed-lag LIO with plane landmarks in the state | VILENS (Wisth et al. RA-L 2021); LIC-Fusion 2.0 (MSCKF, IROS 2020) |
| Plane maps carrying pose uncertainty | VoxelMap RA-L 2022 and its family |
| Registration map positioned as the elevation map | BIEVR-LIO (arXiv Apr 2026) |
| Factor-graph planar regions fed to a planner | Mishra et al. ICRA 2024 (humanoid footsteps, depth camera) |

So the naive port, "range-image Delaunay + geometric checks + ray casting" plus "sparsify the LIO
prior to speed it up", would likely be judged incremental.

### The fact that breaks the SaDVIO "S" story for LiDAR

Every LIO that publishes a runtime breakdown spends its time on data association: KNN or scan-to-map
search, and residual/Jacobian evaluation. The solver and the marginalization are not the cost. This
holds for LiLi-OM, VILENS, FORM (Kaess group, 2025), Super-LIO, and an Orin-NX FAST-LIO2 study. With
planes eliminated (BALM2) the dense prior covers only about 10 × 15 states. **Sparsification sold as
"faster" will be rejected.** It has to be sold as something else (§4, R1).

### The physics that breaks the SaDVIO "D" check for LiDAR

The photometric check works in SaDVIO because a camera sees between landmarks at pixel
resolution, where depth is unknown. LiDAR reflectivity is sampled at **the same pixels as range**,
so a single-scan reflectivity check sees nothing between beams that range does not. Reflectivity
adds information only at:

- mixed pixels and edge bleeding;
- surface orientation, through Lambertian cos θ changes between viewpoints;
- grazing incidence;
- accumulated sub-pixel samples.

A true SaDVIO analogue needs a **second viewpoint**: a later scan ("temporal stereo") or a camera.

---

## 3. Open gaps found (medium confidence unless stated)

| # | Gap | Nearest work to contrast |
|---|---|---|
| G1 | Sparsifying the **fixed-lag marginalization prior of a LIO** (unary KF state + pose-to-plane factors). No hits in web or arXiv-API queries (medium-high). | Hsiung 2018, SaDVIO, SLIM, VILENS |
| G2 | A principled, uncertainty-preserving replacement for **"freeze old scans/planes into the map"**. Voxel-SLAM, LMBAO and submap LIOs condition on old poses, i.e. treat them as having infinite information. | Voxel-SLAM, BALM2, 2026 null-space MSCKF-LIO (Zhu, Huang et al.) |
| G3 | The sliding-window LIO's **own plane/surfel landmarks used directly as the traversability surface**, with uncertainty from the estimator's marginal covariance (including pose correlation), on CPU. | VoxelMap, Mishra ICRA 2024, BIEVR-LIO 2026, Fankhauser 2018 |
| G4 | **Consistency (NEES) of map-element covariance** used for traversability. Not evaluated anywhere found. | STEP RSS 2021, Fankhauser 2018 |
| G5 | **Per-triangle hypothesis test for LiDAR from a second, motion-displaced scan** (range agreement, optionally reflectivity ZNCC through the plane), used to accept or reject surfaces for densification. | COIN-LIO (odometry only), Caraffa 2016, Dold & Brenner 2006 |
| G6 | LiDAR-only, non-learned densification with **calibrated measured-vs-inferred uncertainty**, evaluated downstream on traversability. Also "predict then verify": scoring densified points against later real scans. | Velasco-Sánchez 2023, TULIP, SLAMesh/VDB-GPDF variance |
| G7 | LIO + densification evaluated on **flight-like LiDARs**: narrow FoV and low rate (NASA SQRLi ~40°×40°; an Ames spec of 1 Hz, ≥45°; S3LI's Blickfeld), under a measured CPU/power budget. Almost all work uses 360° spinning sensors. No rover has flown a LiDAR (NASA AAS 25-173). | RoughSense (Sep 2026), ImMesh, SLAMesh |
| G8 | **Dataset**: no public planetary-analog or lava-tube dataset combines rover-height LiDAR, raw IMU with per-point timestamps, GT trajectory, and a dense reference (TLS) with rock/hazard labels (medium-high). | Erfoud, S3LI, Katwijk, TAIL-Plus, LuSNAR (synthetic), CAVERS (karst), Intraleo (no GT) |
| G9 | Effect of LiDAR **degeneracy on the downstream traversability map** (smearing that creates false steps) is unquantified (low-medium). | X-ICP, S3LI (LiDAR SLAM failed there) |
| G10 | Camera ZNCC used as a per-triangle accept/reject test on **LiDAR-derived** triangles (only if a camera is allowed). | FAST-LIVO2 (plane-warped patches, odometry only), ImMesh texturing |

---

## 4. Research options, ranked

### R1 — Unified "landmarks are the terrain" LIO with consistent sparsified marginalization (recommended main paper)

**Claim.** Plane/surfel patches are states in a sliding-window LIO (point-to-plane factors, IMU
preintegration). When a KF leaves, the prior is sparsified (KLD) into a unary KF factor plus
pose-to-plane / absolute-plane factors. Patches therefore leave the window **with their information
intact**, instead of being frozen into the map (Voxel-SLAM) or dropped (SaDVIO drops triangles whose
landmarks were marginalized). The same patches, tessellated, are the traversability surface. The
surface moves when the optimizer moves the patches, and per-cell slope/step uncertainty comes from
the estimator's marginal covariance.

**Why this is the best SaD successor.** It joins S and D in one object, which is SaDVIO's "unified
estimation and reconstruction" idea taken further. It also reframes S from "faster" (not credible
for LIO) to **"consistent, information-preserving marginalization that the map inherits"** (G1, G2,
G3, G4).

**Technical core.**

- Minimal 3-DOF plane chart with anchoring (closest-point form, re-anchored on marginalization as
  in LIC-Fusion 2.0). The stacked topology Jacobian stays square and block-triangular, so the
  SaDVIO closed-form KLD recovery carries over.
- Optional: express the recovered information in BALM2 point-cluster form, at zero extra
  per-iteration cost.
- Marginal covariance recovery restricted to the window; show its cost.
- First-estimate Jacobians (FEJ) / observability so that the uncertainty means something.

**Experiments.**

- NEES of poses and of map elements vs ground truth.
- Accuracy against FAST-LIO2, iG-LIO, DLIO, Point-LIO, Voxel-SLAM and GLIM.
- Dense prior vs sparsified prior vs map freezing: consistency, accuracy, degenerate corridors and
  flat terrain.
- Traversability: map vs dense reference, and calibration of the hazard probabilities.
- Runtime on ARM.

**Reviewer attacks.**

- "VoxelMap + elevation mapping already does this." Answer: estimated states with correlations vs
  propagated, independent covariances; show the NEES difference.
- "Planes are poor on rough terrain." Answer: small patches; test on rock fields.
- "Plane states cost more than BALM elimination." Answer: timing vs Voxel-SLAM.
- "Uncertainty does not change the decision." Answer: a downstream planning or hazard-call
  experiment.

**Competition risk.** The Kaess group (Hsiung 2018, FORM 2025), ETH (BIEVR-LIO 2026) and HKU-MARS
are all close. Move fast or pick a niche such as planetary terrain or flight-like sensors.

**Effort.** High. This is a new back end, though the IMU, Ceres window and KLD machinery are reused.

### R2 — "Temporal stereo" densification for sparse and flight-like LiDAR (most direct D analogue)

**Claim.** Triangles from scan k (range image or tracked patches) are tested against scan k+n, deskewed
and posed by the LIO, the way SaDVIO's right camera tests them:

- rays of the later scan hitting the triangle footprint must agree in range within the noise model;
- optionally, on Ouster, reflectivity ZNCC after the plane-induced warp, as an auxiliary edge /
  mixed-pixel cue;
- optionally, a Lambertian incidence-ratio check on orientation.

Ray casting at finer resolution produces densified points labelled **measured / inferred-verified /
inferred-untested**, each with a calibrated uncertainty (G5, G6).

**Evaluation (makes or breaks it; also covers G7).**

- Ring subsampling of the OS128 (16/32/64) and **flight-like FoV/rate emulation** (40°×40°,
  70°×30°, 1–5 Hz), scored against full-resolution or TLS references.
- Metrics:
  - Chamfer and F-score;
  - Free-Space Violation Ratio (ghost points);
  - per-cell step/slope/roughness error and hazard F1;
  - rock recall vs range;
  - uncertainty calibration (NLL/ECE);
  - ms per scan on one ARM core.
- Baselines: Velasco-Sánchez interpolation, IP-Basic, CURL, SLAMesh, ImMesh, TULIP/ILN (GPU, as an
  upper bound), and FAST-LIO2 + BGK ground filling.

**Reviewer attacks.**

- "Why not just accumulate deskewed scans?" Answer: latency, forward-FoV regions not yet revisited,
  bounded memory.
- Small vertical parallax between consecutive scans on a ground robot, so many triangles cannot be
  tested. Report the untested class honestly.
- "16-beam reflectivity is useless." Keep range as the primary test and reflectivity as an
  ablation only.
- "CURL / OrganizedFastMesh already mesh scans." The contribution is the verification and the
  uncertainty labels, not the meshing.

**Effort.** Medium. This is closest to the existing SaDVIO mesher and the `MarginalDepthInjector`
code. It can run on top of any LIO (FAST-LIO2 at first), independently of R1.

### R3 — Dataset and benchmark (high novelty, high field cost; pairs with R1 or R2)

Extend NUANCES Mars Yard, or run a lava-tube campaign (ExECO context), with:

- OS128 with per-point timestamps and reflectivity;
- raw IMU;
- GNSS/INS ground-truth trajectory;
- **a TLS dense reference**;
- **a rock catalog with heights**;
- hazard labels at GESTALT cell resolution;
- a benchmark protocol for hazard-map accuracy vs compute.

No real lava-tube LiDAR+IMU dataset with a ground-truth trajectory was found (G8). The yard alone
will draw "it's a yard"; a field or lava-tube extension answers that. As a released-data
contribution inside R1/R2 it adds a lot of weight, the same way SaDVIO's dataset did.

### R4 — Same-rig stereo-vs-LiDAR SaD comparison, and SaD-LVIO (only if a camera is allowed)

The Mars-yard rig carries both stereo fisheye and an OS128. That allows a controlled comparison of
when SaDVIO's mesh smooths small rocks and when LiDAR beam sparsity misses them. A hybrid that
**culls LiDAR triangles with camera ZNCC** keeps SaDVIO's original idea, and the camera really does
see between beams (G10). The novelty is moderate-high for the planetary community. The obvious
attack is "FAST-LIVO2/R3LIVE fuse both"; answer it with a hazard-map evaluation under a CPU budget.
This contradicts the "no visual information" premise of Part 4, so it is a fallback.

### R5 — Supporting theory (sections inside R1, not a standalone paper)

- Observability-preserving sparsification: prove the sparsified prior keeps the 4-DOF unobservable
  subspace (with FEJ), in the spirit of Jiang & Shen ICRA 2024.
- KLD-based per-marginalization topology choice: absolute, pose-to-plane, or Chow-Liu over planes.
- Degeneracy to traversability (G9): propagate an X-ICP-style localizability measure into per-cell
  risk (STEP/CVaR style). Needs convincing failure cases.

### Not recommended on their own

- Plain port of KLD sparsification to LIO, sold on speed: thin (see §2).
- Range-image Delaunay + edge/normal filters + ray casting: done (Holz & Behnke, CURL,
  Velasco-Sánchez).
- Single-scan reflectivity ZNCC as the triangle check: physically adds little (see §2).
- Bounded-cost LIO on space-grade CPUs (LEON/HPSC): possibly novel but under-searched, and needs
  target hardware. Keep it as a runtime section, not the claim.

---

## 5. Suggested shape

- **Paper 1 (RA-L style): R1.** S = consistent sparsified marginalization that the map inherits;
  D = landmarks as the terrain surface with estimator uncertainty. Evaluated on the Mars-yard data
  (plus a public LIO benchmark for accuracy) and on flight-like sensor emulation, with the R3
  additions released.
- **Paper 2 (or a workshop/conference paper first, being lower risk): R2**, the verification-based
  densifier, runnable on any LIO. It could be done first because it reuses the most existing code
  and does not depend on the new back end.

R2 first then R1 lowers risk. R1 first is the stronger single claim.

---

## 6. Read in full before writing any novelty sentence

- [ ] CURL (RSS 2022): meshing to upsample a scan
- [ ] Velasco-Sánchez et al., ETFA 2023: CPU range-image interpolation for odometry
- [ ] Holz & Behnke, RAS 2014: organized-grid meshing
- [ ] VILENS (Wisth et al., RA-L 2021): fixed-lag LIO with plane landmarks
- [ ] Voxel-SLAM: how its "marginalization" freezes old scans
- [ ] BALM2: point clusters, plane elimination
- [ ] Zhu, Huang et al., MSCKF-LIO with cluster-to-plane constraints (arXiv 2603.12904, 2026)
- [ ] SLIM: LiDAR map-centric NFR
- [ ] Hsiung et al., IROS 2018 and Jiang & Shen, ICRA 2024
- [ ] FORM (Kaess group, 2025): runtime breakdown and marginalization
- [ ] VoxelMap: plane covariance propagation
- [ ] BIEVR-LIO and COIN-BIEVR (2026)
- [ ] Mishra et al., ICRA 2024: planar regions to planner
- [ ] COIN-LIO: intensity NCC and occlusion check
- [ ] Caraffa et al., ACCV 2016: per-triangle confidence
- [ ] RoughSense (arXiv Sep 2026)
- [ ] MSPA-LIO (could not be read; check how it marginalizes)

## 7. Checks before committing

- [ ] **Dataset contents.** The DOI landing page ("NUANCES - Mars Yard dataset") lists ROS2 bags
      (grand_tour, hills_20, hills_40) and calib.yaml, but no sensor list. Check that the bags
      contain the OS128 with per-point timestamps and reflectivity, the raw IMU, and GT, and which
      sequences from the paper (Rocks, Loops) are present. Check whether any dense reference exists.
- [ ] Manual Google Scholar / IEEE Xplore pass on G1, G3, G5 (the agents could not search Xplore
      full text).
- [ ] Freeze SaDVIO first (step 9 of the VIO fix plan); R1 reuses the IMU and window code.

---

## Sources

### SaDVIO lineage and sparsification theory

- [SaDVIO, RA-L 2024](https://ieeexplore.ieee.org/document/10616232/) · [code](https://github.com/ISAE-PNX/SaDVIO)
- [Debeunne et al., Fast bi-monocular VO using factor graph sparsification, IROS 2023](http://www.iri.upc.edu/files/scidoc/2786-Fast-bi-monocular-visual-odometry-using-factor-graph-sparsification.pdf)
- [NUANCES - Mars Yard dataset](https://dataverse.isae-supaero.fr/citation?persistentId=doi:10.34849/DGLZPE)
- [Hsiung et al., Information Sparsification in VIO, IROS 2018](https://www.cs.cmu.edu/~kaess/pub/Hsiung18iros.pdf)
- [Mazuran et al., Nonlinear Factor Recovery, IJRR 2016](https://journals.sagepub.com/doi/10.1177/0278364915581629)
- [Carlevaris-Bianco et al., Generic Node Removal, T-RO 2014](https://www.semanticscholar.org/paper/Generic-Node-Removal-for-Factor-Graph-SLAM-Carlevaris-Bianco-Kaess/0b9e7c5d54f4f88810ab6666e1b58124eab72c3f)
- [Kretzschmar & Stachniss, Information-theoretic compression of pose graphs for laser SLAM, IJRR 2012](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/kretzschmar12ijrr.pdf)
- [Vallvé et al., Pose-graph sparsification using factor descent, RAS 2019](http://www.iri.upc.edu/files/scidoc/2193-Pose-graph-SLAM-sparsification-using-factor-descent.pdf)
- [Usenko et al., VI mapping with NFR (Basalt), RA-L 2020](https://arxiv.org/abs/1904.06504)
- [Jiang & Shen, Two-step nonlinear factor sparsification, ICRA 2024](https://ieeexplore.ieee.org/document/10610889/)
- [VIO sparsification via Bayes tree, AST 2021](https://www.sciencedirect.com/science/article/abs/pii/S1270963821005757)
- [SLIM, T-RO 2025](https://arxiv.org/abs/2409.08681)

### Sliding-window / fixed-lag LIO and plane landmarks

- [LIO-mapping](https://arxiv.org/abs/1904.06993) · [LiLi-OM](https://arxiv.org/pdf/2010.13150) · [VILENS](https://arxiv.org/abs/2011.06838) · [LIC-Fusion 2.0](https://arxiv.org/abs/2008.07196)
- [MSCKF-LIO with cluster-to-plane constraints (2026)](https://arxiv.org/abs/2603.12904)
- [BALM2](https://arxiv.org/abs/2209.08854) · [BALM](https://arxiv.org/html/2010.08215v2) · [HBA](https://github.com/hku-mars/HBA) · [Voxel-SLAM](https://arxiv.org/abs/2410.08935) · [LMBAO](https://arxiv.org/pdf/2209.08810)
- [π-LSAM](https://ieeexplore.ieee.org/document/9561933/) · [Efficient second-order plane adjustment](https://arxiv.org/abs/2211.11542)
- [GLIM](https://arxiv.org/abs/2407.10344) · [Exact point cloud downsampling, ICRA 2025](https://arxiv.org/abs/2505.01017) · [FORM](https://arxiv.org/html/2510.09966)
- [CT fixed-lag LIC](https://arxiv.org/abs/2302.07456) · [GP-prior radar/lidar-inertial odometry](https://arxiv.org/pdf/2402.06174) · [MSPA-LIO](https://www.researchgate.net/publication/398060895)
- [Kaess, Infinite planes](https://publications.ri.cmu.edu/simultaneous-localization-and-mapping-with-infinite-planes) · [LIPS](https://pgeneva.com/downloads/papers/Geneva2018IROS.pdf) · [Observability of aided INS with points/lines/planes](https://arxiv.org/pdf/1805.05876)
- [KDP-SLAM](https://www.ri.cmu.edu/app/uploads/2017/05/Hsiao17icra.pdf) · [CPA-SLAM](https://cvai.cit.tum.de/_media/spezial/bib/lingni16icra.pdf)

### Filter / efficient LIO baselines and embedded

- [FAST-LIO2](https://arxiv.org/abs/2107.06829) · [Point-LIO](https://github.com/hku-mars/Point-LIO) · [DLIO](https://arxiv.org/abs/2203.03749) · [iG-LIO](https://github.com/zijiechenrobotics/ig_lio) · [VoxelMap](https://arxiv.org/pdf/2109.07082) · [Traj-LO](https://arxiv.org/abs/2309.13842)
- [RESPLE](https://arxiv.org/abs/2504.11580) · [D-LIO](https://arxiv.org/html/2505.16726) · [Super-LIO](https://arxiv.org/pdf/2509.05723) · [QLIO](https://arxiv.org/abs/2503.07949) · [Flight-ready LIO on Orin NX](https://arxiv.org/abs/2607.22145) · [PA-LVIO](https://arxiv.org/abs/2603.16228)
- [Le Gentil, Lisus, Barfoot 2025](https://arxiv.org/abs/2505.04438)

### Meshing, densification, upsampling

- [Gingras et al., CRV 2010](https://www.researchgate.net/publication/224143633_Rough_Terrain_Reconstruction_for_Rover_Motion_Planning) · [Rekleitis et al., AuRo 2013](https://link.springer.com/article/10.1007/s10514-012-9309-9)
- [Holz & Behnke, RAS 2014](https://www.ais.uni-bonn.de/papers/RAS_2014_Holz.pdf) · [Polylidar3D](https://arxiv.org/abs/2007.12065) · [Guinard & Vallet 2018](https://arxiv.org/abs/1804.04001) · [CURL](https://arxiv.org/abs/2205.06059)
- [SLAMesh](https://arxiv.org/abs/2303.05252) · [ImMesh](https://arxiv.org/abs/2301.05206) · [Mesh-LOAM](https://arxiv.org/abs/2312.15630) · [Puma](https://github.com/PRBonn/puma) · [VDBFusion](https://github.com/PRBonn/vdbfusion) · [DB-TSDF](https://arxiv.org/abs/2509.20081)
- [IDTMM](https://ieeexplore.ieee.org/document/10176348/) · [LGFaware-Meshing](https://github.com/Neo-cyber-hubb/LGFaware-Meshing) · [PlanarMesh](https://arxiv.org/abs/2510.13599) · [CAD-Mesher](https://yaepiii.github.io/CAD-Mesher/) · [VDB-GPDF](https://arxiv.org/abs/2407.09649)
- [SHINE-Mapping](https://arxiv.org/abs/2210.02299) · [PIN-SLAM](https://arxiv.org/abs/2401.09101) · [PINGS](https://arxiv.org/abs/2502.05752) · [N3-Mapping](https://arxiv.org/abs/2401.03412) · [NKSR](https://arxiv.org/pdf/2305.19590) · [NeRF-LOAM](https://github.com/JunyuanDeng/NeRF-LOAM)
- [Splat-LOAM](https://arxiv.org/abs/2503.17491) · [Gaussian-LIC](https://arxiv.org/abs/2404.06926) · [Gaussian-LIC2](https://arxiv.org/abs/2507.04004) · [GS-LIVO](https://arxiv.org/abs/2501.08672) · [Real-time LiDAR GS SLAM (2026)](https://arxiv.org/abs/2607.04127) · [SurfFill](https://arxiv.org/abs/2512.03010)
- [Shan et al., LiDAR super-resolution](https://arxiv.org/pdf/2004.05242) · [ILN](https://sgvr.kaist.ac.kr/~yskwon/papers/icra22-iln/ICRA22_iln.pdf) · [TULIP](https://arxiv.org/abs/2312.06733) · [FLASH](https://arxiv.org/abs/2511.07377) · [SRMamba](https://arxiv.org/pdf/2505.10601) · [DU-OR](https://arxiv.org/html/2606.28607)
- [LiDAR SR survey (2026)](https://arxiv.org/html/2602.15904) · [IP-Basic](https://arxiv.org/abs/1802.00036) · [Velasco-Sánchez et al., ETFA 2023](https://ieeexplore.ieee.org/document/10275512/) · [You & Kim 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9823772/)
- [LiDiff](https://github.com/PRBonn/LiDiff) · [LiDPM](https://arxiv.org/pdf/2504.17791) · [Physics-aware diffusion densification (FSVR metric)](https://arxiv.org/html/2603.26759)
- [Caraffa et al., ACCV 2016](https://link.springer.com/chapter/10.1007/978-3-319-54190-7_23) · [Tuley et al., LADAR artifacts](https://www.ri.cmu.edu/pub_files/pub4/tuley_john_2004_1/tuley_john_2004_1.pdf) · [ALICE-LRI](https://arxiv.org/pdf/2510.20708)

### LiDAR intensity / reflectivity

- [Barfoot et al., Into Darkness](https://asrl.utias.utoronto.ca/~jdg/sbib/barfoot_isrr13.pdf) · [COIN-LIO](https://arxiv.org/abs/2310.01235) · [RI-LIO](https://ieeexplore.ieee.org/document/10041769/) · [PG-LIO](https://arxiv.org/abs/2506.18583)
- [COIN-BIEVR](https://icra2026-rigorous-perception.github.io/pdf/pfreundschuh2026.pdf) · [BIEVR-LIO](https://arxiv.org/abs/2604.14421) · [Real-time SLAM with LiDAR intensity](https://arxiv.org/abs/2301.09257) · [Intensity-SLAM](https://arxiv.org/pdf/2102.03798)
- [Levinson & Thrun, beam intensity calibration](http://driving.stanford.edu/papers/ISER2010.pdf) · [Intensity normalization by range and incidence](https://www.isprs.org/proceedings/xxxviii/3-w8/papers/213_laserscanning09.pdf) · [Super LiDAR Intensity](https://arxiv.org/abs/2508.10398)
- [Ouster firmware manual](https://data.ouster.io/downloads/software-user-manual/software-user-manual-v2.1.x.pdf) · [Ye, mixed-pixel removal](https://www.researchgate.net/publication/224327685) · [Dold & Brenner 2006](https://www.isprs.org/proceedings/xxxvi/part5/paper/DOLD_637.pdf)
- [FAST-LIVO2](https://arxiv.org/abs/2408.14035) · [R3LIVE](https://arxiv.org/abs/2109.07982) · [LIV-GaussMap](https://arxiv.org/html/2401.14857v2)

### Unified map/estimator and uncertainty-aware terrain

- [SuMa](https://www.roboticsproceedings.org/rss14/p16.pdf) · [Elastic LiDAR Fusion](https://arxiv.org/abs/1711.01691) · [SLICT](https://arxiv.org/abs/2211.03900) · [MAD-BA](https://arxiv.org/abs/2501.03972) · [2Fast-2Lamaa](https://arxiv.org/pdf/2410.05433)
- [Fankhauser et al., elevation mapping under pose uncertainty](https://github.com/ANYbotics/elevation_mapping) · [GEM](https://github.com/ZJU-Robotics-Lab/GEM) · [Voxgraph](https://arxiv.org/abs/2004.13154) · [Kimera-PGMO](https://github.com/ntnu-arl/kimera_pgmo)
- [Rosinol et al., VIO mesh with regularities](https://arxiv.org/abs/1903.01067) · [Mishra et al., ICRA 2024](https://ieeexplore.ieee.org/document/10610879/) · [PlaneSLAM](https://arxiv.org/abs/2209.08248) · [Mesh navigation](https://github.com/naturerobots/mesh_navigation)

### Traversability, planetary and subterranean, datasets, missions

- [elevation_mapping_cupy](https://arxiv.org/abs/2204.12876) · [STEP](https://arxiv.org/abs/2103.02828) · [TE-NeXt](https://arxiv.org/abs/2406.01395) · [BGK traversability](http://proceedings.mlr.press/v87/shan18a/shan18a.pdf) · [RoughSense](https://arxiv.org/abs/2609.03720) · [PRISM](https://arxiv.org/abs/2607.16366)
- [X-ICP](https://arxiv.org/abs/2211.16335) · [Informed, Constrained, Aligned](https://arxiv.org/abs/2408.11809) · [SubT SLAM survey](https://arxiv.org/abs/2208.01787) · [LIO with DEM constraints for planetary rovers](https://isprs-archives.copernicus.org/articles/XLVIII-3-2024/615/2024/)
- [Erfoud](https://sites.laas.fr/projects/erfoud-dataset/sites.laas.fr/projects/erfoud-dataset/index.html) · [S3LI](https://arxiv.org/abs/2207.06815) · [S3LI Vulcano](https://arxiv.org/abs/2601.19557) · [Katwijk](https://journals.sagepub.com/doi/10.1177/0278364917737153) · [TAIL](https://arxiv.org/abs/2403.16875) · [LuSNAR](https://arxiv.org/abs/2407.06512) · [CAVERS](https://arxiv.org/abs/2604.15052) · [Intraleo lava tube](https://pmc.ncbi.nlm.nih.gov/articles/PMC13454644/)
- [NASA lunar surface nav survey AAS 25-173](https://ntrs.nasa.gov/api/citations/20250000720/downloads/Lunar_Surface_Nav_Survey_AAS_GNC_011825.pdf) · [SQRLi tech brief](https://ntts-prod.s3.amazonaws.com/t2p/prod/t2media/tops/pdf/GSC-TOPS-352.pdf) · [DAEDALUS](https://activities.esa.int/4000130925) · [KNaCK](https://ntrs.nasa.gov/citations/20230012518)
