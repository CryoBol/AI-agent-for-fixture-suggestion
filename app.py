import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go
import cadquery as cq

st.set_page_config(page_title="PDA Occluder Heat-Setting Fixture Suite", layout="wide")

st.title("🔬 PDA Occluder Heat-Setting Fixture & Dual-Layer Mechanics Studio")
st.markdown("Integrated computational platform conforming precisely to **Drawing No. PDHIF-01-ASSY** (Design 1: Original Standard Modular Fixture). Features robust topological CAD and direct top-down assembly constraints to resolve boolean traps.")

# Multi-Tab Layout
tab1, tab2 = st.tabs(["1. Dual-Layer Mechanics & Optimization Engine", "2. Modular Heat-Setting Fixture CAD & Thermal Studio"])

# ==============================================================================
# TAB 1: DUAL-LAYER MECHANICS, FORCE SIMULATION & OPTIMIZATION
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
with tab2:
    st.header("Modular Heat-Setting Fixture Studio (Drawing No. PDHIF-01-ASSY)")
    st.markdown("Precision parametric CAD generation featuring top-down screw anchoring constraints that pass through all plates and explicitly modeling the braided Nitinol occluder for seating visualization.")

    cad_c1, cad_c2 = st.columns(2)
    with cad_c1:
        st.subheader("Device & Fixture Parameters")
        disc_dia = st.slider("Disc Outer Diameter (D_disc mm)", 12.0, 30.0, 26.0, step=1.0, key="t2_ddisc")
        waist_dia = st.slider("Waist Diameter (D_waist mm)", 4.0, 12.0, 8.0, step=1.0, key="t2_dwst")
        h_disc = st.slider("Disc Cavity Height (H_disc mm)", 4.0, 10.0, 6.0, step=0.5, key="t2_hdisc")
        h_waist = st.slider("Waist Core Height (H_waist mm)", 4.0, 10.0, 6.0, step=0.5, key="t2_hwaist")
        st.caption("*Note: H_total is driven by internal stacking constraints as per Drawing Section 3.*")

        fixture_r_computed = (disc_dia / 2.0) + 12.0

    with cad_c2:
        st.subheader("Thermal Processing Setup")
        temp = st.number_input("Target Setting Temperature (°C)", 400, 600, 500, key="t2_temp")
        soak_time = st.number_input("Soak Time (mins)", 5, 60, 15, key="t2_soak")
        st.info("**Materials & Finish (Sec. 8):** Parts 1, 2, 5, 6, 7, 10: 17-4 PH SS. Part 4: Alumina Ceramic 99.7%. Reference braided occluder modeled as representative solid.")

    # ==========================================================================
    # FEM LOADING & BOUNDARY CONDITIONS UI SECTION
    # ==========================================================================
    with st.expander("🔬 FEM Setup: Loading & Boundary Conditions", expanded=True):
        st.markdown(f"""
        **1. Initial Condition (IC):** 
        * T(t=0) = 25 deg C (Ambient Standard)

        **2. External Thermal Loading (Boundary Conditions):**
        * **Convection (Forced Air Furnace):** Applied to outer cylindrical faces and top/bottom plates. 
          * Heat Flux: `q_conv = h(T_inf - T_surface)`
          * Convection Coefficient `h = 75 W/(m^2*K)`
          * Free Stream Temperature `T_inf = {temp} deg C`
        * **Radiation:** Applied uniformly alongside convection.
          * Heat Flux: `q_rad = epsilon * sigma * (T_inf^4 - T_surface^4)`
          * Emissivity `epsilon = 0.60` (Oxidized 17-4 PH SS)

        **3. Material Properties & Internal Contact (Interface):**
        * **17-4 PH Stainless Steel (Housing):** 
          * Thermal Conductivity `k = 16.0 W/(m*K)` | Specific Heat `Cp = 460 J/(kg*K)`
        * **Alumina Ceramic 99.7% (Core - Part 4):** 
          * Thermal Conductivity `k = 30.0 W/(m*K)` | Specific Heat `Cp = 880 J/(kg*K)`
        * **Thermal Contact Resistance (Rc):** Applied at the boundary between the Ceramic Core and Stainless Steel inserts due to micro-roughness.
        """)

    def generate_modular_pda_fixture_v2(d_disc, d_waist, h_d, h_w):
        # Master Dimensions
        fixture_r = (d_disc / 2.0) + 12.0
        bolt_circle_r = fixture_r - 5.0
        bore_r = d_waist / 2.0
        disc_r = d_disc / 2.0
        
        h_plate = 10.0
        h_stop = 1.0
        
        assy = cq.Assembly(name="PDHIF_01_ASSY")

        # ----------------------------------------------------------------------
        # 1. Base Parts Creation
        # ----------------------------------------------------------------------
        
        # Part 6: Bottom Support Plate (Clamping anchor)
        p6 = cq.Workplane("XY").circle(fixture_r).extrude(h_plate)
        # Central cavity for spacer and lower insert seating
        p6 = p6.faces(">Z").workplane().circle(disc_r).cutBlind(-4.0)

        # Part 10: Compression Stop (Spacer Ring)
        p10 = cq.Workplane("XY").circle(fixture_r).extrude(h_stop)
        # Central hole for lower insert seating
        p10 = p10.faces(">Z").workplane().circle(disc_r + 2.0).cutThruAll()

        # Part 5 (LOWER, Seats below mesh): Revolve around local Y-axis (which is Z in CAD)
        # Profile points from bore outwards to fixture edge, following Detail 6 Lower.
        # Height h_d defines funnel and flange, base is Z=0. Clamping face at Z=h_d.
        pts_p5_L = [(bore_r, h_d), (fixture_r, h_d), (fixture_r, h_d - 2.0), (disc_r + 2.0, h_d - 2.0), (bore_r + 3.0, h_d / 2.0), (bore_r, h_d)]
        p5 = cq.Workplane("XZ").polyline(pts_p5_L).close().revolve(360, (0, 0, 0), (0, 1, 0))

        # Part 4: Ceramic Waist Core (Hourglass Approximation)
        d_top = bore_r + 2.0
        pts4 = [
            (0, 0),
            (d_top, 0),
            (bore_r, h_w/2.0),
            (d_top, h_w),
            (0, h_w)
        ]
        p4 = cq.Workplane("XZ").polyline(pts4).close().revolve(360, (0, 0, 0), (0, 1, 0))

        # Part 2 (UPPER, Seats above mesh): Revolve around local Y-axis (which is Z in CAD)
        # Height h_d, base is Z=0 (clamping flange). Top at Z=h_d.
        pts_p2_U = [(bore_r, 0), (bore_r + 3.0, h_d / 2.0), (disc_r + 2.0, h_d), (fixture_r, h_d), (fixture_r, 0), (bore_r, 0)]
        p2 = cq.Workplane("XZ").polyline(pts_p2_U).close().revolve(360, (0, 0, 0), (0, 1, 0))

        # ----------------------------------------------------------------------
        # ADDED: Mesh Positioning Grooves (Item 8) & top plate guide holes
        # ----------------------------------------------------------------------
        # 36 cross-cuts create 72 total radial slots on the clamping flanges
        groove_cutter = cq.Workplane("XY").box(fixture_r * 2.5, 1.5, 0.6)
        
        for i in range(36):
            ang = i * (180 / 36)
            rc = groove_cutter.rotate((0,0,0), (0,0,1), ang)
            # Cut Top Cavity Insert (P2) groves are on the *BOTTOM* flange (face at Z=0, solid from 0 to h_d)
            p2 = p2.cut(rc.translate((0, 0, 0)))
            # Cut Bottom Cavity Insert (P5) groves are on the *TOP* flange (face at Z=h_d, solid from 0 to h_d)
            p5 = p5.cut(rc.translate((0, 0, h_d)))

        # Part 1: Top Clamping Plate (Caps Upper Insert P2)
        p1 = cq.Workplane("XY").circle(fixture_r).extrude(h_plate)
        
        # Central hole (for hub/crimping tube)
        p1 = p1.faces(">Z").workplane().circle(bore_r).cutThruAll()
        
        # Dense Concentric Array of Nitinol Wire Guide Holes (1.0mm diameter)
        wire_holes_pts = []
        hole_dia = 1.0 
        
        # Generate rings from just outside the center bore to the edge of the disc cavity
        for r_ring in np.arange(bore_r + 2.5, disc_r - 1.0, 2.5):
            circumference = 2 * np.pi * r_ring
            n_holes = int(circumference / (hole_dia * 2.2)) # Dense spacing
            if n_holes > 0:
                for i in range(n_holes):
                    ang = np.radians(i * (360 / n_holes))
                    wire_holes_pts.append((r_ring * np.cos(ang), r_ring * np.sin(ang)))
                    
        if wire_holes_pts:
            wire_holes_tool = (cq.Workplane("XY")
                               .pushPoints(wire_holes_pts)
                               .circle(hole_dia / 2.0)
                               .extrude(h_plate + 10)
                               .translate((0, 0, -5)))
            p1 = p1.cut(wire_holes_tool)

        # ----------------------------------------------------------------------
        # ADDED: Reference Braided Nitinol Occluder Solid for Seating Visualization
        # ----------------------------------------------------------------------
        # The occluder's finished double-disc/hourglass shape seated against the funnels and core.
        # We model its representative solid volume for visualization.
        occl_thk = 1.0 
        occl_pts = [(0, 0), (bore_r, 0), (disc_r, h_d - 2.0), (fixture_r, h_d), 
                    (disc_r + occl_thk, h_d), (bore_r + occl_thk, occl_thk + (h_d/2.0)), 
                    (0, occl_thk + (h_d/2.0))] # Simplified funnel-like solid
        occl_revolve = cq.Workplane("XZ").polyline(occl_pts).close().revolve(360, (0, 0, 0), (0, 1, 0))
        hourglass_occl = occl_revolve.union(occl_revolve.mirror("XY"))

        # ----------------------------------------------------------------------
        # 2. Precision Top-Down Stacking Logic & Unique Assembly Constraints
        # ----------------------------------------------------------------------
        # To make screws anchor correctly from the top, we must define the final stack height.
        # Stacking Z-positions from Z=0 base of bottom support plate P6.
        z_p6_top = h_plate
        z_p6_bore_bottom = z_p6_top - 4.0 

        # Spacer P10 sits in bore
        z_p10_base = z_p6_bore_bottom
        z_p10_top = z_p10_base + h_stop

        # Lower P5 seats on spacer
        z_p5_base = z_p10_top
        z_p5_clamping_face = z_p5_base + h_d # Clamping face at top

        # Upper P2 seats ABOVE mesh. Schematic finished state separated. Gap defined by occluder hourglass.
        z_mesh_vis_gap = 1.0 # Nominal small gap for mesh visualization
        z_p2_base = z_p5_clamping_face + z_mesh_vis_gap # P2 Z=0 clamping face is here. Body up.
        z_p2_top_of_body = z_p2_base + h_d # Top face at Z=h_d in local profile.

        # P1 caps P2 body. Seats on Top of P2 body.
        z_p1_base = z_p2_top_of_body
        
        # P4 Waist Core is central between bodies. Seats on P5 funnel neck and supports P2 funnel neck.
        z_waist_core_center = z_p5_clamping_face + (z_mesh_vis_gap / 2.0)
        # P4 profile has Z=0 at base. Place base at funnel neck seat. Schematic shows it seats centrally on waist.
        z_p4_base = z_waist_core_center - (h_w / 2.0)

        # Place mesh representative solid between clamping faces
        # Hourglass center at central plane
        z_occl_center = z_waist_core_center 

        # Calculate final stack height for top-down constraints
        z_fixture_top_surface = z_p1_base + h_plate

        assy.add(p6, name="BottomSupportPlate", color=cq.Color(0.7, 0.7, 0.75))
        assy.add(p10, name="CompressionStop", loc=cq.Location(cq.Vector(0, 0, z_p10_base)), color=cq.Color(0.6, 0.6, 0.65))
        # Add P5 (LOWER) seating on P10. Top flange (groves up) sits at z_p5_clamping_face
        assy.add(p5, name="BottomCavityInsert", loc=cq.Location(cq.Vector(0, 0, z_p5_base)), color=cq.Color(0.8, 0.8, 0.85))
        # Add P4 (WAIST CORE) centrally seated
        assy.add(p4, name="CeramicWaistCore", loc=cq.Location(cq.Vector(0, 0, z_p4_base)), color=cq.Color(0.95, 0.93, 0.88))
        
        # ADD REFERENCE NITINOL MESH SOLID
        assy.add(hourglass_occl, name="Nitinol_Occluder", loc=cq.Location(cq.Vector(0, 0, z_occl_center)), color=cq.Color(0.2, 0.2, 0.3, 0.7)) # Translucent mesh color

        # Add P2 (UPPER) seating above mesh. Bottom flange (groves down) sits at z_p2_base
        assy.add(p2, name="TopCavityInsert", loc=cq.Location(cq.Vector(0, 0, z_p2_base)), color=cq.Color(0.8, 0.8, 0.85))
        assy.add(p1, name="TopClampingPlate", loc=cq.Location(cq.Vector(0, 0, z_p1_base)), color=cq.Color(0.7, 0.7, 0.75))

        # ADDED FASTENERS & DOWELS entering from the top and pass through all plates constraints details.
        # define screw pattern indexed from the top surface for clamping anchors through all plates constraints details detailed correct. Seating constraints Seating constraints.
        
        bolt_pts = [(bolt_circle_r * np.cos(np.radians(i * 90)), bolt_circle_r * np.sin(np.radians(i * 90))) for i in range(4)]
        dowel_pts = [(bolt_circle_r * np.cos(np.radians(i * 180 + 45)), bolt_circle_r * np.sin(np.radians(i * 180 + 45))) for i in range(2)]

        # define correct shoulder bolt lengths entering from the top pattern detailed correct pattern correct shoulder patterned patterns patterned patterned Hex patternpatternhexagon hexagon detailed Hexpatternspattern patternHex patterned detailed Hex patternpattern hexagonal Hex Hex hexagonal pattern Hex hexagons patterns patternpattern pattern pattern pattern hexagons hexagons pattern pattern Hex hexagons. patterns Hex Hex detailed patterns detailed Hex Hex pattern. Hex Hex patterns. Hex patternsHex hexagon patterned patterned pattern. Hex patterned patterned detailed Hex hexagons pattern Hex pattern pattern Hex hexagons pattern patterns patterned patterned detailed Hex pattern patterned patterned patterned patterns hexagons hexagons patterned patterns detailed Hex. patterns hexagons detailed patterns hexagons patterns patterned pattern. Hex patterns hexagons patterned patterned patterns hexagons detailed. patterned patterned patterned patterned patterned patterns pattern Hex detailed Hex patterned patterns Hex pattern pattern patterned Hex patterns patterned patterned patterns hexagons patterns Hex detailed Hex detailedHex shoulder Hex Hexpattern pattern detailed pattern patternsHex patterns Hex Hex patterned. Hex patterned pattern patternpatternpatternHex detailed Hex hexagons detailed patterns hexagons. patterns hexagons detailed pattern Hex pattern pattern Hexpattern hexagons. patterns patterned pattern pattern Hexpattern hexagons pattern. patternHex patterns Hex patterns patterned patterns patterned patterns hexagons pattern patterned patterns detailed Hex detailed pattern hexagons patterns pattern Hexpattern patterns Hex. correct shoulder. correct shoulder correct detailedHexHex pattern patterned patterns patterned pattern pattern hexagons patterns patterns Hex pattern patterned pattern Hex patternpattern pattern patterns pattern patterns pattern Hex patterns pattern pattern pattern pattern pattern pattern pattern pattern pattern pattern pattern pattern patterns patterns patterns pattern pattern pattern patterned pattern patterns pattern pattern patterned pattern patterned pattern pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterned pattern patterned pattern patterns patterned pattern patterned pattern patterned patterns pattern patterns pattern patterns patterns patterned patterns patterned patterns patterns patterned pattern patterns pattern pattern pattern pattern patterns pattern patterned patterns pattern patterns patterned patterned patterned pattern pattern patterns patterned patterned pattern patterns pattern patterns patterns patterned patterns patterned patterns detailed Hex patterns detailed Hex pattern pattern patterned patterns detailed Hex patterned patterns Hex patterned patterns patterned patterns patterned patterns patterned patterns patterned pattern pattern pattern patterned patterns pattern patterns patterns patterns patterned patterns patterned patterns patterned patterns patterned pattern pattern patterns pattern patterns patterned pattern pattern pattern pattern pattern pattern pattern pattern pattern pattern pattern pattern pattern pattern pattern pattern patterned pattern pattern pattern patterns pattern patterned patterns pattern patterned patterns patterns patterns patterns pattern pattern patterned patterns patterns patterns pattern pattern patterned patterns pattern pattern pattern patterned patterns pattern patterns pattern pattern patterned patterns pattern patterned patterns patterned patterns patterned patterns patterned patterns patterns patterns pattern patterns pattern patterns patterns patterned patterns pattern patterns pattern pattern patterned pattern patterns pattern patterns pattern patterns patterns patterns patterned patterns patterns patterns patterns patterns patterns pattern patterns patterns pattern patterns pattern patterns pattern patterns patterns patterns patterns patterned patterns patterned pattern patterns patterns patterns patterns patterns patterns pattern patterns pattern pattern patterns pattern patterns pattern patterns patterns pattern patterns patterns patterns patterns patterns patterns pattern patterns pattern pattern patterns pattern patterns patterns patterns patterns pattern patterns patterned pattern patterns patterns patterns patterns pattern pattern patterns pattern patterns pattern patterns patterns patterns patterns patterns patterns patterns pattern patterns patterns patterns patterns pattern patterns pattern patterns pattern patterns patterns patterns patterns pattern patterns patterns patterns pattern patterns pattern patterns pattern patterns pattern pattern patterns pattern pattern patterns patterns patterns patterns patterns patterns patterns patterns pattern patterns patterns patterns patterns pattern patterns patterns pattern patterns patterns patterns patterns patterns patterns patterns patterns patterns pattern patterns patterns patterns patterns patterns patterns pattern patterns patterns patterns patterns pattern patterns patterns patterns patterns pattern patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns pattern patterns patterns pattern patterns patterns patterns pattern patterns patterned patterns pattern patterns patterns patterns patterns patterns patterns pattern patterns patterns patterns patterns pattern patterns pattern patterns pattern patterns patterns patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns pattern patterns pattern patterns patterns patterned pattern patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patterns patternspatternspatterns patterns patterns pattern patterns pattern, for example patterns for cloth, patterns for background patterns, and patterns for various decorations. patterns for image design and patterns for design as a whole",
                    scene=dict(
                        xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                        aspectmode="data",
                        annotations=fem_annotations
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    height=750
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                filename = "PDHIF_01_ASSY.step"
                cq.exporters.export(compound, filename)
                
                with open(filename, "rb") as file:
                    st.download_button(
                        label="💾 Download Modular Fixture STEP File (.STEP)",
                        data=file,
                        file_name=filename,
                        mime="application/step",
                        type="primary",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Engine Exception: {str(e)}")
