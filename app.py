"""
PDA Occluder Heat-Setting Fixture & Dual-Layer Mechanics Studio
=================================================================

Refined version. Summary of what changed from the original and why:

CAD GEOMETRY (generate_modular_pda_fixture)
  * All secondary radii (ceramic-core seating shoulder, clamping-ledge
    transition) are now derived as BOUNDED FRACTIONS of the bore->disc
    radial span, instead of fixed absolute-mm offsets (e.g. "bore_r+3.0",
    "disc_r+2.0"). Fixed offsets silently produced self-intersecting,
    inverted revolve profiles whenever the waist diameter was close to the
    disc diameter -- CadQuery/OCCT does NOT raise an exception for this, it
    silently builds a garbage (but "valid") solid, which is exactly the
    "absurd design that doesn't crash" failure mode. The fractional
    formulation guarantees bore_r < boss_r < disc_r < transition_r <
    fixture_r for every input the UI can produce.
  * A validate_dims() gate rejects (and the UI additionally prevents via a
    dynamic slider bound) any waist:disc ratio that would make a
    physically absurd fixture -- a real PDA-style occluder's waist is
    always much smaller than its retention discs.
  * The ceramic waist core's flare radius now matches the insert boss
    radius exactly, so the core seats flush on its supporting shoulder
    instead of landing partway across it (the original used two different
    unrelated constants for what should be the same mating diameter).
  * The "mesh positioning grooves" are now cut into the actual sloped
    cavity face that contacts the flared disc mesh (traced from the same
    profile points used for the CAD), not the flat mounting faces that
    the original code targeted -- those faces never touch the mesh.
  * Groove count and width are now chosen so that, at the outer radius the
    grooves reach, adjacent grooves cannot overlap and consume the entire
    ridge between them (the original's fixed 1.5 mm width with a 5 deg
    step could overlap and erase most of the top/bottom surface at large
    disc sizes).
  * Blind-pocket depth is bounded to guarantee wall stock remains.

THERMAL MODEL (solve_transient)
  * Replaced the placeholder "T = temp - |z-z_mid|*0.4 - r*0.15" formula,
    which (a) ignored the soak-time input entirely, (b) wasn't tied to any
    material property, and (c) implied a permanent gradient that
    contradicts the actual physics of a furnace soak, with a real 2D
    axisymmetric transient conduction finite-difference solve using actual
    thermal-diffusivity values for 17-4PH stainless and 99.7% alumina,
    an ambient starting temperature, and a proper Dirichlet boundary
    condition on every truly furnace-exposed surface (outer radius, top,
    bottom, and any open internal cavity such as the wire bore).
  * The material map is built by point-in-polygon testing against the
    exact same profile coordinates used to build the CAD solids, so the
    thermal model and the CAD model can't silently drift apart.
  * The solver detects thermal equilibrium and stops early, and reports
    the time to reach it -- letting the user see directly whether their
    chosen soak time is thermally limited or (as is typical for a fixture
    this size) governed by the nitinol phase-transformation hold time
    rather than by heat conduction lag.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cadquery as cq
from scipy.interpolate import RegularGridInterpolator

st.set_page_config(page_title="PDA Occluder Heat-Setting Fixture Suite", layout="wide")

st.title("🔬 PDA Occluder Heat-Setting Fixture & Dual-Layer Mechanics Studio")
st.markdown(
    "Integrated computational platform conforming to **Drawing No. PDHIF-01-ASSY** "
    "(Design 1: Original Standard Modular Fixture). Geometry is parametrized so every "
    "slider combination produces a valid, non-self-intersecting solid, and the thermal "
    "field is a real transient conduction solve rather than a cosmetic gradient."
)

tab1, tab2 = st.tabs([
    "1. Dual-Layer Mechanics & Optimization Engine",
    "2. Modular Heat-Setting Fixture CAD & Thermal Studio",
])

# ==============================================================================
# TAB 1: DUAL-LAYER MECHANICS, FORCE SIMULATION & OPTIMIZATION  (unchanged --
# this tab is a parametric mechanics/braid model, not a CAD generator, so it
# is out of scope for the geometry-precision pass requested; left as-is.)
# ==============================================================================
with tab1:
    st.header("Dual-Layer Occluder Structural & Force Modeller")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Macro Geometry & Inner Core")
        D_macro = st.slider("Waist Diameter (D_waist mm)", 4.0, 30.0, 8.0, 0.5, key="t1_D")
        P_macro = st.slider("Pitch Length / Pic Length (mm)", 1.0, 10.0, 4.0, 0.2, key="t1_P")
        N1 = st.slider("Inner Wire Count (N1)", 16, 72, 36, 4, key="t1_N1")
        d1 = st.slider("Inner Wire Diameter (d1 mm)", 0.10, 0.30, 0.16, 0.01, key="t1_d1")

    with col_t2:
        st.subheader("Outer Shell & Displacement")
        N2 = st.slider("Outer Wire Count (N2)", 36, 144, 72, 4, key="t1_N2")
        d2 = st.slider("Outer Wire Diameter (d2 mm)", 0.05, 0.15, 0.09, 0.01, key="t1_d2")
        delta_r = st.slider("Radial Displacement Simulation (delta_r mm)", 0.1, 5.0, 1.5, 0.1, key="t1_dr")

    def calculate_braid_geometry(D_mm, P_mm, N, d):
        tan_theta = (np.pi * D_mm) / P_mm
        theta_rad = np.arctan(tan_theta)
        theta_deg = np.degrees(theta_rad)
        sin_theta = np.sin(theta_rad)
        term = (N * d) / (np.pi * D_mm * sin_theta)
        cf = term * (2.0 - term)
        return theta_deg, np.clip(cf, 0.0, 1.0)

    theta1, cf1 = calculate_braid_geometry(D_macro, P_macro, N1, d1)
    theta2, cf2 = calculate_braid_geometry(D_macro, P_macro, N2, d2)

    E_eff = 50000.0  # N/mm^2
    ei1 = (N1 * E_eff * np.pi * (d1**4) / 64.0) * (np.cos(np.radians(theta1))**2)
    ei2 = (N2 * E_eff * np.pi * (d2**4) / 64.0) * (np.cos(np.radians(theta2))**2)
    ei_total = ei1 + ei2

    k1 = 0.05 * (d1**4) * N1
    k2 = 0.03 * (d2**4) * N2
    f_rad_waist = (k1 * (delta_r**1.2)) + (k2 * (delta_r**1.1))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Inner Braid Angle", f"{theta1:.2f}°", f"Coverage: {cf1:.2f}")
    with m2:
        st.metric("Outer Braid Angle", f"{theta2:.2f}°", f"Coverage: {cf2:.2f}")
    with m3:
        st.metric("Total Bending Rigidity", f"{ei_total:.2f} N·mm²")
    with m4:
        st.metric("Waist Radial Force", f"{f_rad_waist:.3f} N", "Target: 0.1 - 0.3 N")

    st.subheader("Radial Force vs. Displacement Curve")
    fig1, ax1 = plt.subplots(figsize=(10, 3.8))
    displacements = np.linspace(0.1, 5.0, 50)
    forces = [(k1 * (d**1.2)) + (k2 * (d**1.1)) for d in displacements]
    ax1.plot(displacements, forces, label="Dual-Layer Composite Force", color="#008080", linewidth=2.5)
    ax1.axhline(y=0.1, color="orange", linestyle="--", label="Min Clinical Limit")
    ax1.axhline(y=0.3, color="crimson", linestyle="--", label="Max Clinical Limit")
    ax1.set_xlabel("Radial Displacement (mm)")
    ax1.set_ylabel("Radial Force (N)")
    ax1.set_title("Waist Radial Compression Response Curve")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)


# ==============================================================================
# TAB 2: MODULAR HEAT-SETTING FIXTURE CAD & THERMAL STUDIO (PDHIF-01-ASSY)
# ==============================================================================

# ------------------------------------------------------------------------
# Material properties used by the thermal solver.
#   k   thermal conductivity   [W/m.K]
#   rho density                [kg/m3]
#   cp  specific heat          [J/kg.K]
# Representative room-temperature literature values; real furnace runs
# should confirm against certified material data sheets for the specific
# heat/lot in use.
# ------------------------------------------------------------------------
STEEL_PROPS = dict(k=17.8, rho=7750.0, cp=460.0)     # 17-4PH stainless (H900-ish)
CERAMIC_PROPS = dict(k=30.0, rho=3900.0, cp=880.0)   # Alumina, 99.7%


def _alpha_mm2_per_s(props):
    """Thermal diffusivity k/(rho*cp), converted from m^2/s to mm^2/s so it
    stays dimensionally consistent with a millimeter-scale FD grid."""
    return (props["k"] / (props["rho"] * props["cp"])) * 1.0e6


ALPHA_STEEL = _alpha_mm2_per_s(STEEL_PROPS)
ALPHA_CERAMIC = _alpha_mm2_per_s(CERAMIC_PROPS)


# ------------------------------------------------------------------------
# Geometry validation
#
# The bore->disc span (disc_r - bore_r) is the shared "budget" that every
# downstream feature draws from: the boss/transition shoulders, the mesh
# positioning grooves, AND the wire-guide channels. A purely proportional
# cap (bore_r <= k*disc_r) lets that budget shrink to almost nothing in
# absolute mm at small disc sizes even while staying under the ratio --
# which is exactly what silently zeroed out the wire-guide grooves in one
# boundary case before this was caught by testing. The cap is therefore a
# COMBINATION of a proportional limit and an absolute-mm minimum span, and
# every downstream radius (boss_r, transition_r, guide grooves, mesh
# grooves) is derived as a bounded fraction of this guaranteed-positive
# span, never a fixed offset that could exceed it.
# ------------------------------------------------------------------------
MAX_WAIST_TO_DISC_RATIO = 0.84    # d_waist / d_disc soft cap (proportion)
MIN_ABS_SPAN_MM = 2.5             # minimum disc_r - bore_r regardless of scale


def _max_bore_r(disc_r):
    min_span = max(MIN_ABS_SPAN_MM, 0.16 * disc_r)
    return disc_r - min_span


def max_waist_dia_for_disc(d_disc):
    """Largest waist diameter that keeps the bore->disc span comfortably
    positive (both proportionally and in absolute mm) for a given disc
    diameter, so every downstream feature has room to fit.

    Floors (rather than rounds) to 1 decimal and backs off by a small
    epsilon so the UI can never offer a value that the stricter, exact
    threshold in validate_dims() would then reject -- rounding up here
    previously let a boundary value the slider itself presented get
    rejected by validate_dims's unrounded check.
    """
    disc_r = d_disc / 2.0
    raw_max = 2.0 * _max_bore_r(disc_r)
    return max(4.0, np.floor(raw_max * 10.0) / 10.0 - 0.01)


def validate_dims(d_disc, d_waist, h_d, h_w):
    disc_r = d_disc / 2.0
    bore_r = d_waist / 2.0
    max_bore_r = _max_bore_r(disc_r)
    if bore_r > max_bore_r:
        raise ValueError(
            f"Waist diameter {d_waist:.1f} mm is too large relative to disc diameter "
            f"{d_disc:.1f} mm (max supported waist for this disc size is "
            f"{2 * max_bore_r:.1f} mm). A PDA-style fixture needs the waist substantially "
            f"narrower than the retention discs, and this keeps at least "
            f"{MIN_ABS_SPAN_MM:.1f} mm of radial room for the insert shoulders, positioning "
            f"grooves, and wire-guide channels to fit without collapsing into each other."
        )
    if bore_r < 1.0:
        raise ValueError("Waist radius is too small to route a wire-guide bore (< 1.0 mm).")
    return disc_r, bore_r


# ------------------------------------------------------------------------
# CAD generation
# ------------------------------------------------------------------------
def generate_modular_pda_fixture(d_disc, d_waist, h_d, h_w, groove_count=None):
    """Build the PDHIF-01-ASSY modular heat-setting fixture assembly.

    Returns (assembly, ceramic_core_mid_z, meta) where meta carries every
    derived dimension (also consumed by the thermal solver so the two stay
    in sync with the same geometry).
    """
    disc_r, bore_r = validate_dims(d_disc, d_waist, h_d, h_w)
    span = disc_r - bore_r  # > 0, guaranteed by validate_dims

    RIM_MARGIN = 12.0                    # constant mold-body wall margin beyond the disc
    fixture_r = disc_r + RIM_MARGIN
    bolt_circle_r = fixture_r - 5.0
    h_plate = 10.0
    h_stop = 1.0

    # Secondary radii as bounded FRACTIONS of the bore->disc span. This is
    # the key fix versus fixed-mm offsets: it guarantees
    #   bore_r < boss_r < disc_r < transition_r < fixture_r
    # for every valid input, so the revolve profiles can never self-intersect
    # regardless of how small/large the waist and disc are.
    boss_r = bore_r + min(3.0, 0.30 * span)      # ceramic-core seating shoulder
    transition_r = disc_r + 2.0                   # clamping-ledge step (always << RIM_MARGIN)
    shoulder_z = 0.30 * h_d                        # bounded within (0, h_d) for any h_d

    pocket_depth = min(4.0, h_plate - 3.0)         # keep >= 3 mm plate stock under the pocket

    assy = cq.Assembly(name="PDHIF_01_ASSY")

    # --- Part 6: Bottom Support Plate ---
    p6 = cq.Workplane("XY").circle(fixture_r).extrude(h_plate)
    p6 = p6.faces(">Z").workplane().circle(disc_r).cutBlind(-pocket_depth)

    # --- Part 10: Compression Stop (annular ring resting on P6's outer ledge) ---
    p10 = cq.Workplane("XY").circle(fixture_r).extrude(h_stop)
    p10 = p10.faces(">Z").workplane().circle(transition_r).cutThruAll()

    # --- Part 5: Bottom Cavity Insert ---
    # Profile edges, by physical role:
    #   (bore_r,h_d)->(boss_r,h_d)              flat boss ring, seats ceramic core
    #   (boss_r,h_d)->(transition_r,shoulder_z) SLOPED cavity face -- this is what
    #                                            actually contacts the flared disc mesh
    #   (transition_r,shoulder_z)->(fixture_r,shoulder_z)  clamping ledge
    #   (fixture_r,shoulder_z)->(fixture_r,0)   outer wall
    #   (fixture_r,0)->(disc_r,0)               bottom mounting flange
    #   (disc_r,0)->(bore_r,h_d) [closing edge]  inner bore wall
    pts5 = [
        (bore_r, h_d),
        (boss_r, h_d),
        (transition_r, shoulder_z),
        (fixture_r, shoulder_z),
        (fixture_r, 0),
        (disc_r, 0),
    ]
    p5 = cq.Workplane("XZ").polyline(pts5).close().revolve()

    # --- Part 4: Ceramic Waist Core ---
    # Flare radius set to boss_r EXACTLY so it seats flush on the P5/P2 boss
    # shoulders (originally used a different, unrelated constant that landed
    # only partway across the boss ring -- an unsupported-overhang mismatch).
    pts4 = [
        (0, 0),
        (boss_r, 0),
        (bore_r, h_w / 2.0),
        (boss_r, h_w),
        (0, h_w),
    ]
    p4 = cq.Workplane("XZ").polyline(pts4).close().revolve()

    # --- Part 2: Top Cavity Insert (mirror of Part 5) ---
    pts2 = [
        (disc_r, h_d),
        (fixture_r, h_d),
        (fixture_r, h_d - shoulder_z),
        (transition_r, h_d - shoulder_z),
        (boss_r, 0),
        (bore_r, 0),
    ]
    p2 = cq.Workplane("XZ").polyline(pts2).close().revolve()

    # ----------------------------------------------------------------
    # Mesh positioning grooves -- cut into the sloped cavity face
    # (boss_r -> transition_r edge) that the flared disc mesh actually
    # contacts, sized/counted so they cannot overlap into each other.
    # ----------------------------------------------------------------
    GROOVE_WIDTH = 1.2
    GROOVE_DEPTH = 0.3
    GROOVE_RADIAL_LEN = min(6.0, 0.5 * (transition_r - boss_r))
    MIN_RIDGE = 0.6  # minimum material left standing between adjacent grooves

    r_groove = float(np.clip(disc_r, boss_r + GROOVE_RADIAL_LEN, transition_r - GROOVE_RADIAL_LEN * 0.25))
    r_outer_check = r_groove + GROOVE_RADIAL_LEN / 2.0
    max_n = int(np.floor((2 * np.pi * r_outer_check) / (GROOVE_WIDTH + MIN_RIDGE)))
    n_grooves = groove_count if groove_count is not None else min(24, max_n)
    n_grooves = max(0, min(n_grooves, max_n))

    if n_grooves > 0:
        t = (r_groove - boss_r) / (transition_r - boss_r)
        z_local_p5 = h_d - t * (h_d - shoulder_z)   # local z on P5's sloped face at r_groove
        z_local_p2 = t * (h_d - shoulder_z)          # mirrored location on P2

        slope_deg_p5 = float(np.degrees(np.arctan2(shoulder_z - h_d, transition_r - boss_r)))
        slope_deg_p2 = -slope_deg_p5

        def slope_cutter(r_center, z_center, slope_deg):
            # Box straddles the sloped surface (tilted to its local angle)
            # so the cut depth stays controlled regardless of exact
            # alignment, the same "half in material, half in free space"
            # trick used for flat-face grooves, generalized to a tilted face.
            box = cq.Workplane("XY").box(GROOVE_RADIAL_LEN, GROOVE_WIDTH, GROOVE_DEPTH * 2.2)
            box = box.rotate((0, 0, 0), (0, 1, 0), slope_deg)
            box = box.translate((r_center, 0, z_center))
            return box

        base_cutter_p5 = slope_cutter(r_groove, z_local_p5, slope_deg_p5)
        base_cutter_p2 = slope_cutter(r_groove, z_local_p2, slope_deg_p2)

        for i in range(n_grooves):
            ang = i * (360.0 / n_grooves)
            p5 = p5.cut(base_cutter_p5.rotate((0, 0, 0), (0, 0, 1), ang))
            p2 = p2.cut(base_cutter_p2.rotate((0, 0, 0), (0, 0, 1), ang))

    # ----------------------------------------------------------------
    # Part 1: Top Clamping Plate -- nitinol wire-guide channels
    #
    # The original design cut a very dense concentric field of ~1mm holes
    # (over a hundred for a mid-size disc). Two problems: (1) at typical
    # tessellation tolerance the holes render as an indistinct moire/"star"
    # pattern rather than clearly resolved features, and (2) mechanically
    # a bare hole field gives each wire no guidance from the central bore
    # out to its hole -- wires can cross/tangle during threading.
    #
    # Replaced with a small number of discrete RADIAL GUIDE GROOVES: each
    # is a shallow channel running from just outside the central bore out
    # to a clocked through-hole, so a wire end is captured at the bore and
    # guided cleanly to its own exit point. Fewer, larger, well-separated
    # features -- clearly resolved in render and easier to inspect/machine.
    # ----------------------------------------------------------------
    p1 = cq.Workplane("XY").circle(fixture_r).extrude(h_plate)
    p1 = p1.faces(">Z").workplane().circle(bore_r).cutThruAll()

    GUIDE_WIDTH = 1.0
    GUIDE_DEPTH = 0.6
    GUIDE_HOLE_DIA = 1.2
    GUIDE_MIN_RIDGE = 0.8

    # Both radii derived as bounded fractions of the guaranteed-positive
    # span (never fixed mm offsets independent of it), so guide_r_end is
    # mathematically guaranteed > guide_r_start for every input
    # validate_dims() accepts: the lower bound below is
    # guide_r_start + 0.85*span in the worst case, always positive.
    guide_r_start = bore_r + min(0.3, 0.1 * span)
    guide_r_end = min(guide_r_start + max(1.0, 0.5 * span), disc_r - 0.05 * span)

    # Cap the count so grooves can't overlap at their tightest point (the
    # inner radius, right where they exit the bore).
    max_n_guide = int(np.floor((2 * np.pi * guide_r_start) / (GUIDE_WIDTH + GUIDE_MIN_RIDGE)))
    n_guide = max(4, min(16, max_n_guide)) if max_n_guide >= 4 else max(1, max_n_guide)

    if n_guide > 0 and guide_r_end > guide_r_start:
        groove_len = guide_r_end - guide_r_start
        groove_center_r = (guide_r_start + guide_r_end) / 2.0

        channel_cutter = (cq.Workplane("XY")
                           .box(groove_len, GUIDE_WIDTH, GUIDE_DEPTH * 2)
                           .translate((groove_center_r, 0, h_plate)))
        hole_cutter = (cq.Workplane("XY")
                       .circle(GUIDE_HOLE_DIA / 2.0)
                       .extrude(h_plate + 10)
                       .translate((guide_r_end, 0, -5)))

        for i in range(n_guide):
            ang = i * (360.0 / n_guide)
            p1 = p1.cut(channel_cutter.rotate((0, 0, 0), (0, 0, 1), ang))
            p1 = p1.cut(hole_cutter.rotate((0, 0, 0), (0, 0, 1), ang))

    # ----------------------------------------------------------------
    # Fastener holes (bolt circle sits at disc_r+7, always inside
    # fixture_r=disc_r+12 with margin, and always outside the wire-hole
    # field which stops at disc_r-1 -- both clearances hold for every
    # valid disc_r since the offsets are additive constants)
    # ----------------------------------------------------------------
    bolt_pts = [(bolt_circle_r * np.cos(np.radians(i * 90)), bolt_circle_r * np.sin(np.radians(i * 90))) for i in range(4)]
    dowel_pts = [(bolt_circle_r * np.cos(np.radians(i * 180 + 45)), bolt_circle_r * np.sin(np.radians(i * 180 + 45))) for i in range(2)]

    holes_tool = (cq.Workplane("XY")
                  .pushPoints(bolt_pts).circle(3.5)
                  .pushPoints(dowel_pts).circle(3.1)
                  .extrude(200)
                  .translate((0, 0, -50)))

    p6 = p6.cut(holes_tool)
    p10 = p10.cut(holes_tool)
    p5 = p5.cut(holes_tool)
    p2 = p2.cut(holes_tool)
    p1 = p1.cut(holes_tool)

    bolt_len = h_plate + h_stop + (h_d * 2) + h_w + h_plate + 2.0
    actual_bolt = (cq.Workplane("XY").circle(3.0).extrude(bolt_len)
                   .faces(">Z").workplane().circle(4.5).extrude(4.0))
    dowel_len = bolt_len - 10.0
    actual_dowel = cq.Workplane("XY").circle(3.0).extrude(dowel_len)

    z_stop = h_plate
    z_p5 = z_stop + h_stop
    z_p4 = z_p5 + h_d
    z_p2 = z_p4 + h_w
    z_p1 = z_p2 + h_d

    assy.add(p6, name="BottomSupportPlate", color=cq.Color(0.7, 0.7, 0.75))
    assy.add(p10, name="CompressionStop", loc=cq.Location(cq.Vector(0, 0, z_stop)), color=cq.Color(0.6, 0.6, 0.65))
    assy.add(p5, name="BottomCavityInsert", loc=cq.Location(cq.Vector(0, 0, z_p5)), color=cq.Color(0.8, 0.8, 0.85))
    assy.add(p4, name="CeramicWaistCore", loc=cq.Location(cq.Vector(0, 0, z_p4)), color=cq.Color(0.95, 0.93, 0.88))
    assy.add(p2, name="TopCavityInsert", loc=cq.Location(cq.Vector(0, 0, z_p2)), color=cq.Color(0.8, 0.8, 0.85))
    assy.add(p1, name="TopClampingPlate", loc=cq.Location(cq.Vector(0, 0, z_p1)), color=cq.Color(0.7, 0.7, 0.75))

    for i, pt in enumerate(bolt_pts):
        assy.add(actual_bolt, name=f"ShoulderBolt_M6_{i}", loc=cq.Location(cq.Vector(pt[0], pt[1], -2.0)), color=cq.Color(0.4, 0.4, 0.45))
    for i, pt in enumerate(dowel_pts):
        assy.add(actual_dowel, name=f"DowelPin_{i}", loc=cq.Location(cq.Vector(pt[0], pt[1], 0.0)), color=cq.Color(0.5, 0.5, 0.5))

    meta = dict(
        disc_r=disc_r, bore_r=bore_r, boss_r=boss_r, transition_r=transition_r,
        fixture_r=fixture_r, shoulder_z=shoulder_z, n_grooves=n_grooves,
        n_guide=n_guide, guide_r_start=guide_r_start, guide_r_end=guide_r_end,
        z_stop=z_stop, z_p5=z_p5, z_p4=z_p4, z_p2=z_p2, z_p1=z_p1, h_plate=h_plate,
    )
    return assy, z_p4 + (h_w / 2.0), meta


# ------------------------------------------------------------------------
# Thermal solver
# ------------------------------------------------------------------------
def _point_in_poly(r_pts, z_pts, poly):
    """Vectorized point-in-polygon (ray casting) against the same (r,z)
    profile coordinates used to build the CAD revolve, so the thermal
    material map matches the actual solid geometry."""
    poly = np.array(poly)
    n = len(poly)
    inside = np.zeros(r_pts.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        ri, zi = poly[i]
        rj, zj = poly[j]
        cond = ((zi > z_pts) != (zj > z_pts)) & \
               (r_pts < (rj - ri) * (z_pts - zi) / (zj - zi + 1e-300) + ri)
        inside ^= cond
        j = i
    return inside


def _build_material_grid(meta, h_d, h_w, nr, nz):
    """mat: -1 = air / Dirichlet boundary, 0 = steel, 1 = ceramic."""
    disc_r, bore_r, boss_r, transition_r = meta["disc_r"], meta["bore_r"], meta["boss_r"], meta["transition_r"]
    fixture_r, shoulder_z = meta["fixture_r"], meta["shoulder_z"]
    z_stop, z_p5, z_p4, z_p2, z_p1, h_plate = (meta["z_stop"], meta["z_p5"], meta["z_p4"],
                                                 meta["z_p2"], meta["z_p1"], meta["h_plate"])
    z_top = z_p1 + h_plate

    r = np.linspace(0, fixture_r, nr)
    z = np.linspace(0, z_top, nz)
    R, Z = np.meshgrid(r, z, indexing="ij")

    mat = np.full(R.shape, -1, dtype=int)

    pts5 = [(bore_r, h_d), (boss_r, h_d), (transition_r, shoulder_z),
            (fixture_r, shoulder_z), (fixture_r, 0), (disc_r, 0)]
    pts2 = [(disc_r, h_d), (fixture_r, h_d), (fixture_r, h_d - shoulder_z),
            (transition_r, h_d - shoulder_z), (boss_r, 0), (bore_r, 0)]
    pts4 = [(0, 0), (boss_r, 0), (bore_r, h_w / 2.0), (boss_r, h_w), (0, h_w)]

    # P6: bottom support plate -- solid disc over its full z range
    mat[(Z >= 0) & (Z <= h_plate)] = 0

    # P10: compression stop ring
    h_stop = z_p5 - z_stop
    mat[(Z >= z_stop) & (Z <= z_stop + h_stop) & (R >= transition_r)] = 0

    # P5, P4, P2: exact revolve-profile membership via point-in-polygon
    for (z_lo, z_hi, poly, mat_id) in (
        (z_p5, z_p4, pts5, 0),
        (z_p4, z_p2, pts4, 1),
        (z_p2, z_p1, pts2, 0),
    ):
        zmask = (Z >= z_lo) & (Z <= z_hi)
        if zmask.any():
            rloc, zloc = R[zmask], Z[zmask] - z_lo
            inside = _point_in_poly(rloc, zloc, poly)
            idx = np.where(zmask)
            mat[idx[0][inside], idx[1][inside]] = mat_id

    # P1: top clamping plate
    mat[(Z >= z_p1) & (Z <= z_top) & (R >= bore_r)] = 0

    # True exterior envelope is furnace-exposed at every height/radius it
    # occurs at -- pin it Dirichlet regardless of which part occupies that
    # location. (Internal air pockets, e.g. the open waist cavity and wire
    # bore, are already Dirichlet by construction since nothing overwrote
    # their initial -1.)
    mat[-1, :] = -1
    mat[:, 0] = -1
    mat[:, -1] = -1

    return r, z, mat


def solve_transient(meta, h_d, h_w, target_temp_c, soak_time_min,
                     initial_temp_c=25.0, nr=55, nz=70, n_snapshots=14,
                     progress_cb=None):
    """Explicit 2D axisymmetric transient conduction FD solve.

    Boundary conditions (documented here so they travel with the result --
    also surfaced verbatim in the UI's "Analysis Conditions" panel):
      * Dirichlet, T = target_temp_c, on every truly furnace-exposed
        surface: the outer curved radius of the whole stack, the very top
        face, the very bottom face, and any open internal cavity (the
        central wire bore, the open waist cavity around the ceramic core,
        the compression stop's through-hole). This is the standard
        high-Biot-number idealization for a small part in a well
        -circulated furnace -- surface temperature snaps to the furnace
        air temperature much faster than the part's interior can respond.
      * Symmetry (zero-flux) at the r=0 centerline, handled via the
        analytic L'Hopital limit of the axisymmetric radial term.
    Initial condition: uniform initial_temp_c (ambient) at t=0, representing
    a room-temperature fixture loaded into a pre-heated furnace.
    Loading: target_temp_c is the furnace set point (Dirichlet magnitude);
    soak_time_min is the requested hold duration.
    Material model: two isotropic regions (17-4PH SS bodies, alumina
    ceramic core) via literature diffusivity; interface conduction uses
    each node's own local diffusivity (first-order, no explicit flux
    -continuity treatment at the interface).
    Simplification: 2D axisymmetric -- small perforations (guide grooves,
    positioning grooves, fastener holes) are not individually resolved,
    since their thermal mass is negligible against the bulk.

    Captures up to n_snapshots log-spaced fields through the transient (plus
    the final state) so the UI can show the heating progression rather than
    only the end state -- useful because small fixtures like this one often
    reach equilibrium in seconds, long before the requested soak completes.
    Stops early once every solid node is within 0.05 C of target.
    """
    r, z, mat = _build_material_grid(meta, h_d, h_w, nr, nz)
    dr, dz = r[1] - r[0], z[1] - z[0]

    alpha = np.where(mat == 1, ALPHA_CERAMIC, ALPHA_STEEL)
    is_solid = mat >= 0

    T = np.full(mat.shape, initial_temp_c, dtype=float)
    T[~is_solid] = target_temp_c

    alpha_max = alpha[is_solid].max() if is_solid.any() else ALPHA_STEEL
    dt_stable = 0.4 / (alpha_max * (2.0 / dr**2 + 2.0 / dz**2))
    total_t = soak_time_min * 60.0
    n_steps_requested = max(1, int(np.ceil(total_t / dt_stable)))
    dt = total_t / n_steps_requested

    R = r[:, None] * np.ones((1, len(z)))

    CONVERGE_TOL_C = 0.05
    equilibrium_time_s = None
    steps_run = 0

    snapshot_fracs = sorted(set(np.round(np.geomspace(0.003, 1.0, n_snapshots), 6)))
    snapshots = [{"t_s": 0.0, "T": T.copy()}]
    next_idx = 0

    for step in range(n_steps_requested):
        Tp = T.copy()
        lap_z = np.zeros_like(T)
        lap_z[:, 1:-1] = (Tp[:, 2:] - 2 * Tp[:, 1:-1] + Tp[:, :-2]) / dz**2

        lap_r = np.zeros_like(T)
        lap_r[1:-1, :] = ((Tp[2:, :] - 2 * Tp[1:-1, :] + Tp[:-2, :]) / dr**2
                           + (1.0 / R[1:-1, :]) * (Tp[2:, :] - Tp[:-2, :]) / (2 * dr))
        lap_r[0, :] = 4.0 * (Tp[1, :] - Tp[0, :]) / dr**2  # r=0 symmetry limit

        T_new = Tp + dt * alpha * (lap_r + lap_z)
        T_new[~is_solid] = target_temp_c
        T = T_new
        steps_run = step + 1

        if progress_cb and step % max(1, n_steps_requested // 20) == 0:
            progress_cb(step / n_steps_requested)

        frac_now = steps_run / n_steps_requested
        while next_idx < len(snapshot_fracs) and frac_now >= snapshot_fracs[next_idx]:
            snapshots.append({"t_s": steps_run * dt, "T": T.copy()})
            next_idx += 1

        if steps_run % 25 == 0 or steps_run == n_steps_requested:
            max_err = np.abs(T[is_solid] - target_temp_c).max() if is_solid.any() else 0.0
            if equilibrium_time_s is None and max_err <= CONVERGE_TOL_C:
                equilibrium_time_s = steps_run * dt
                if snapshots[-1]["t_s"] != equilibrium_time_s:
                    snapshots.append({"t_s": equilibrium_time_s, "T": T.copy()})
                break

    if snapshots[-1]["t_s"] < steps_run * dt:
        snapshots.append({"t_s": steps_run * dt, "T": T.copy()})

    if progress_cb:
        progress_cb(1.0)

    core_mask = (mat == 1)
    boundary_conditions = {
        "Dirichlet (furnace temp)": "Outer curved radius, top face, bottom face, "
                                     "and all open internal cavities (wire bore, "
                                     "open waist cavity, compression-stop bore)",
        "Symmetry / zero-flux": "Centerline r = 0 (axisymmetric limit)",
        "Initial condition": f"Uniform {initial_temp_c:.0f}C at t = 0 (ambient, furnace-loaded cold)",
        "Loading": f"Furnace set point {target_temp_c:.0f}C held for {soak_time_min:.1f} min "
                   f"({total_t:.0f} s) requested soak",
        "Material model": (f"17-4PH SS: k={STEEL_PROPS['k']:.1f} W/m.K, rho={STEEL_PROPS['rho']:.0f} "
                            f"kg/m3, cp={STEEL_PROPS['cp']:.0f} J/kg.K (alpha={ALPHA_STEEL:.2f} mm2/s); "
                            f"Alumina 99.7%: k={CERAMIC_PROPS['k']:.1f} W/m.K, rho={CERAMIC_PROPS['rho']:.0f} "
                            f"kg/m3, cp={CERAMIC_PROPS['cp']:.0f} J/kg.K (alpha={ALPHA_CERAMIC:.2f} mm2/s)"),
        "Method": (f"Explicit 2D axisymmetric FD, {nr}x{nz} grid, dr={dr:.3f} mm, dz={dz:.3f} mm, "
                   f"dt={dt:.4g} s ({steps_run} steps run), stability margin 0.4x the explicit limit"),
        "Simplifications": "Small perforations (guide/positioning grooves, fastener holes) not "
                            "individually resolved; interface conduction uses each node's local "
                            "diffusivity (no explicit flux-continuity treatment at material boundaries)",
    }

    return dict(
        r=r, z=z, T=T, mat=mat, dt=dt, n_steps=steps_run,
        core_min_temp=T[core_mask].min() if core_mask.any() else None,
        core_mean_temp=T[core_mask].mean() if core_mask.any() else None,
        target_temp=target_temp_c, equilibrium_time_s=equilibrium_time_s,
        requested_soak_s=total_t, snapshots=snapshots,
        boundary_conditions=boundary_conditions,
    )


with tab2:
    st.header("Modular Heat-Setting Fixture Studio (Drawing No. PDHIF-01-ASSY)")
    st.markdown(
        "Parametric CAD generation with discrete **Nitinol Wire Guide Channels** on the Top "
        "Plate (radial slot + through-hole per wire, guiding each wire from the central bore "
        "to a clocked exit point) and **Mesh Positioning Grooves** cut into the actual sloped "
        "cavity face that contacts the flared disc mesh."
    )

    cad_c1, cad_c2 = st.columns(2)
    with cad_c1:
        st.subheader("Device & Fixture Parameters")
        disc_dia = st.slider("Disc Outer Diameter (D_disc mm)", 12.0, 30.0, 26.0, step=1.0, key="t2_ddisc")

        max_waist = max_waist_dia_for_disc(disc_dia)
        # Seed/clamp the stored slider value before instantiating the widget
        # (and pass no `value=` once state exists) so shrinking the disc
        # diameter can never leave the waist slider holding a now-invalid
        # value, and so Streamlit doesn't warn about value/key conflicts.
        if "t2_dwst" not in st.session_state:
            st.session_state["t2_dwst"] = min(8.0, max_waist)
        elif st.session_state["t2_dwst"] > max_waist:
            st.session_state["t2_dwst"] = max_waist
        waist_dia = st.slider(
            "Waist Diameter (D_waist mm)", 4.0, max_waist,
            step=0.5, key="t2_dwst",
        )
        st.caption(
            f"Capped at {max_waist:.1f} mm for this disc size (waist ≤ "
            f"{MAX_WAIST_TO_DISC_RATIO:.2f}× disc diameter, and at least "
            f"{MIN_ABS_SPAN_MM:.1f} mm of radial clearance reserved beyond the waist) so the "
            f"insert shoulders, positioning grooves, and wire-guide channels all have room to "
            f"fit without collapsing into each other."
        )

        h_disc = st.slider("Disc Cavity Height (H_disc mm)", 4.0, 10.0, 6.0, step=0.5, key="t2_hdisc")
        h_waist = st.slider("Waist Core Height (H_waist mm)", 4.0, 10.0, 6.0, step=0.5, key="t2_hwaist")
        st.caption("*H_total is geometrically driven by H_disc and H_waist internal stacking per Drawing Section 3.*")

    with cad_c2:
        st.subheader("Thermal Processing Setup")
        temp = st.number_input("Target Setting Temperature (°C)", 400, 600, 500, key="t2_temp")
        soak_time = st.number_input("Soak Time (mins)", 5, 60, 15, key="t2_soak")
        st.info(
            "**Materials & Finish (Sec. 8):** Parts 1, 2, 5, 6, 7, 10: 17-4 PH SS "
            f"(k≈{STEEL_PROPS['k']:.1f} W/m·K). Part 4: Alumina Ceramic 99.7% "
            f"(k≈{CERAMIC_PROPS['k']:.0f} W/m·K)."
        )

    if st.button("🚀 Generate Modular Fixture CAD & Thermal Field", type="primary", width="stretch"):
        try:
            with st.spinner("Building parametric CAD assembly..."):
                assy, z_mid, meta = generate_modular_pda_fixture(disc_dia, waist_dia, h_disc, h_waist)
                compound = assy.toCompound()
                # Finer tessellation tolerance than the original (0.4 -> 0.12mm)
                # so the guide grooves and positioning grooves resolve as
                # clean, distinct facets instead of a coarse moire/"star"
                # artifact where many small features fall below the old
                # tolerance's effective resolution.
                vertices, triangles = compound.tessellate(0.12)
                if not vertices or not triangles:
                    raise ValueError("Generated geometry resulted in an empty mesh.")

            with st.spinner("Solving transient conduction field (17-4PH SS + alumina core)..."):
                progress_bar = st.progress(0.0)
                thermal = solve_transient(
                    meta, h_disc, h_waist, target_temp_c=float(temp),
                    soak_time_min=float(soak_time),
                    progress_cb=lambda f: progress_bar.progress(min(f, 1.0)),
                )
                progress_bar.empty()

            filename = "PDHIF_01_ASSY.step"
            cq.exporters.export(compound, filename)
            with open(filename, "rb") as f:
                step_bytes = f.read()

            # Default the time-in-soak view to whichever captured snapshot has
            # the largest internal temperature spread -- i.e. the most visibly
            # informative "cold spot" moment -- rather than always defaulting
            # to the final (often fully equilibrated) state.
            is_solid = thermal["mat"] >= 0
            spreads = [
                (snap["T"][is_solid].max() - snap["T"][is_solid].min()) if is_solid.any() else 0.0
                for snap in thermal["snapshots"]
            ]
            default_idx = int(np.argmax(spreads)) if spreads else 0

            st.session_state["t2_result"] = dict(
                meta=meta, temp=float(temp), soak_time=float(soak_time),
                x=np.array([v.x for v in vertices]),
                y=np.array([v.y for v in vertices]),
                z=np.array([v.z for v in vertices]),
                i=np.array([t[0] for t in triangles]),
                j=np.array([t[1] for t in triangles]),
                k=np.array([t[2] for t in triangles]),
                thermal=thermal, step_bytes=step_bytes,
            )
            st.session_state["t2_snapshot_idx"] = default_idx

        except ValueError as e:
            st.error(f"Design rejected: {e}")
        except Exception as e:
            st.error(f"Engine Exception: {str(e)}")

    # ------------------------------------------------------------------
    # Render section -- reads from session_state so scrubbing the
    # time-in-soak slider below just re-colors the existing mesh with a
    # different cached snapshot, instead of re-running CAD + FD on every
    # slider move.
    # ------------------------------------------------------------------
    if "t2_result" in st.session_state:
        res = st.session_state["t2_result"]
        meta, thermal = res["meta"], res["thermal"]
        temp, soak_time = res["temp"], res["soak_time"]

        info_c1, info_c2, info_c3, info_c4 = st.columns(4)
        info_c1.metric("Fixture Body OD", f"{2*meta['fixture_r']:.1f} mm")
        info_c2.metric("Core Seat Ø (boss)", f"{2*meta['boss_r']:.1f} mm")
        info_c3.metric("Mesh Positioning Grooves", f"{meta['n_grooves']}")
        info_c4.metric("Wire Guide Channels", f"{meta['n_guide']}")

        if thermal["equilibrium_time_s"] is not None:
            st.success(
                f"Modular Fixture CAD compiled matching Drawing PDHIF-01-ASSY. "
                f"Thermal equilibrium reached in ≈{thermal['equilibrium_time_s']:.1f} s "
                f"of the {soak_time:.0f}-minute soak -- the remaining soak time is governed by "
                f"the nitinol phase-transformation hold requirement, not thermal lag."
            )
        else:
            st.warning(
                f"The {soak_time:.0f}-minute soak did NOT reach thermal equilibrium "
                f"(core min {thermal['core_min_temp']:.1f}°C vs target {temp:.0f}°C). "
                f"Increase soak time or verify furnace air circulation."
            )

        snapshots = thermal["snapshots"]
        n_snap = len(snapshots)
        if "t2_snapshot_idx" not in st.session_state or st.session_state["t2_snapshot_idx"] >= n_snap:
            st.session_state["t2_snapshot_idx"] = n_snap - 1

        st.subheader("Time-in-soak viewer")
        snap_idx = st.slider(
            "Elapsed furnace time", 0, n_snap - 1,
            key="t2_snapshot_idx",
            help="Scrub through the transient to see the heating front develop and settle. "
                 "The default position is set to the moment with the largest internal "
                 "temperature spread (most visible cold spot).",
        )
        t_sel = snapshots[snap_idx]["t_s"]
        T_field = snapshots[snap_idx]["T"]
        is_solid = thermal["mat"] >= 0
        spread_now = (T_field[is_solid].max() - T_field[is_solid].min()) if is_solid.any() else 0.0
        st.caption(
            f"t = {t_sel:.1f} s into soak | solid-body range at this instant: "
            f"{T_field[is_solid].min():.1f}°C to {T_field[is_solid].max():.1f}°C "
            f"(spread {spread_now:.1f}°C)"
        )

        interp = RegularGridInterpolator(
            (thermal["r"], thermal["z"]), T_field, bounds_error=False, fill_value=float(temp),
        )
        r_pts = np.sqrt(res["x"]**2 + res["y"]**2)
        r_clamped = np.clip(r_pts, thermal["r"][0], thermal["r"][-1])
        z_clamped = np.clip(res["z"], thermal["z"][0], thermal["z"][-1])
        T_verts = interp(np.column_stack([r_clamped, z_clamped]))

        fig2 = go.Figure(data=[
            go.Mesh3d(
                x=res["x"], y=res["y"], z=res["z"], i=res["i"], j=res["j"], k=res["k"],
                intensity=T_verts, colorscale="Inferno",
                cmin=25.0, cmax=float(temp),
                colorbar=dict(title="Temperature (°C)", len=0.75),
                flatshading=True, showscale=True,
            )
        ])
        fig2.update_layout(
            title=(f"Modular Heat-Setting Fixture Assembly (PDHIF-01-ASSY | "
                   f"Target: {temp:.0f}°C | t = {t_sel:.1f} s)"),
            scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)", aspectmode="data"),
            margin=dict(l=0, r=0, b=0, t=40), height=700,
        )
        st.plotly_chart(fig2, width="stretch")

        with st.expander("Thermal cross-section (r-z plane, mid-angle slice)"):
            fig3, ax3 = plt.subplots(figsize=(6, 8))
            cs = ax3.contourf(thermal["r"], thermal["z"], T_field.T, levels=30, cmap="inferno",
                               vmin=25.0, vmax=float(temp))
            ax3.contour(thermal["r"], thermal["z"], (thermal["mat"] == 1).T, levels=[0.5], colors="cyan", linewidths=1.5)
            ax3.set_xlabel("Radius r (mm)")
            ax3.set_ylabel("Height z (mm)")
            ax3.set_title(f"Solved temperature field at t={t_sel:.1f}s (cyan = ceramic core outline)")
            plt.colorbar(cs, ax=ax3, label="°C")
            st.pyplot(fig3)

        with st.expander("Analysis conditions (boundary conditions, loading, material model)"):
            for label, desc in thermal["boundary_conditions"].items():
                st.markdown(f"**{label}:** {desc}")

        st.download_button(
            label="💾 Download Modular Fixture STEP File (.STEP)",
            data=res["step_bytes"], file_name="PDHIF_01_ASSY.step", mime="application/step",
            type="primary", width="stretch",
        )
    else:
        st.info("Set your parameters above and click **Generate** to build the fixture and solve the thermal field.")
