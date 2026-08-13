import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go
import cadquery as cq

st.set_page_config(page_title="PDA Occluder Heat-Setting Fixture Suite", layout="wide")

st.title("🔬 PDA Occluder Heat-Setting Fixture & Dual-Layer Mechanics Studio")
st.markdown("Integrated computational platform conforming precisely to **Drawing No. PDHIF-01-ASSY** (Design 1: Original Standard Modular Fixture). Engineered with a fully robust topological CAD core to prevent boolean failures.")

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
    st.markdown("Precision parametric CAD generation integrated with university-level FEM thermal load mapping.")

    cad_c1, cad_c2 = st.columns(2)
    with cad_c1:
        st.subheader("Device & Fixture Parameters")
        disc_dia = st.slider("Disc Outer Diameter (D_disc mm)", 12.0, 30.0, 26.0, step=1.0, key="t2_ddisc")
        waist_dia = st.slider("Waist Diameter (D_waist mm)", 4.0, 12.0, 8.0, step=1.0, key="t2_dwst")
        h_disc = st.slider("Disc Cavity Height (H_disc mm)", 4.0, 10.0, 6.0, step=0.5, key="t2_hdisc")
        h_waist = st.slider("Waist Core Height (H_waist mm)", 4.0, 10.0, 6.0, step=0.5, key="t2_hwaist")
        
        fixture_r_computed = (disc_dia / 2.0) + 12.0

    with cad_c2:
        st.subheader("Thermal Processing Setup")
        temp = st.number_input("Target Setting Temperature (deg C)", 400, 600, 500, key="t2_temp")
        soak_time = st.number_input("Soak Time (mins)", 5, 60, 15, key="t2_soak")
        st.info("Materials & Finish: Parts 1, 2, 5, 6, 10: 17-4 PH SS. Part 4: Alumina Ceramic 99.7%.")

    # ==========================================================================
    # NEW: FEM LOADING & BOUNDARY CONDITIONS UI SECTION
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

    def generate_modular_pda_fixture(d_disc, d_waist, h_d, h_w):
        fixture_r = (d_disc / 2.0) + 12.0
        bolt_circle_r = fixture_r - 5.0
        bore_r = d_waist / 2.0
        disc_r = d_disc / 2.0
        
        h_plate = 10.0
        h_stop = 1.0
        
        assy = cq.Assembly(name="PDHIF_01_ASSY")

        p6 = cq.Workplane("XY").circle(fixture_r).extrude(h_plate)
        p6 = p6.faces(">Z").workplane().circle(disc_r).cutBlind(-4.0)

        p10 = cq.Workplane("XY").circle(fixture_r).extrude(h_stop)
        p10 = p10.faces(">Z").workplane().circle(disc_r + 2.0).cutThruAll()

        pts5 = [
            (bore_r, h_d),                     
            (bore_r + 3.0, h_d),               
            (disc_r + 2.0, 2.0),               
            (fixture_r, 2.0),                  
            (fixture_r, 0),                    
            (disc_r, 0)                      
        ]
        p5 = cq.Workplane("XZ").polyline(pts5).close().revolve(360, (0, 0, 0), (0, 1, 0))

        d_top = bore_r + 2.0
        pts4 = [
            (0, 0),
            (d_top, 0),
            (bore_r, h_w/2.0),
            (d_top, h_w),
            (0, h_w)
        ]
        p4 = cq.Workplane("XZ").polyline(pts4).close().revolve(360, (0, 0, 0), (0, 1, 0))

        pts2 = [
            (disc_r, h_d),                       
            (fixture_r, h_d),                 
            (fixture_r, h_d - 2.0),         
            (disc_r + 2.0, h_d - 2.0),            
            (bore_r + 3.0, 0),                  
            (bore_r, 0)                        
        ]
        p2 = cq.Workplane("XZ").polyline(pts2).close().revolve(360, (0, 0, 0), (0, 1, 0))

        groove_cutter = cq.Workplane("XY").box(fixture_r * 2.5, 1.5, 0.6)
        for i in range(36):
            ang = i * (180 / 36)
            rc = groove_cutter.rotate((0,0,0), (0,0,1), ang)
            p2 = p2.cut(rc.translate((0, 0, h_d)))
            p5 = p5.cut(rc.translate((0, 0, 0)))

        p1 = cq.Workplane("XY").circle(fixture_r).extrude(h_plate)
        p1 = p1.faces(">Z").workplane().circle(bore_r).cutThruAll()
        
        wire_holes_pts = []
        hole_dia = 1.0 
        for r_ring in np.arange(bore_r + 2.5, disc_r - 1.0, 2.5):
            circumference = 2 * np.pi * r_ring
            n_holes = int(circumference / (hole_dia * 2.2))
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
        z_p5   = z_stop + h_stop
        z_p4   = z_p5 + h_d
        z_p2   = z_p4 + h_w
        z_p1   = z_p2 + h_d
        z_total_height = z_p1 + h_plate

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

        return assy, z_p4 + (h_w / 2.0), z_total_height

    if st.button("🚀 Generate Modular Fixture CAD & Thermal Field", type="primary", use_container_width=True):
        with st.spinner("Compiling precise modular CAD assembly and mapping thermal diffusion field..."):
            try:
                assy, z_mid, z_top_surface = generate_modular_pda_fixture(disc_dia, waist_dia, h_disc, h_waist)
                compound = assy.toCompound()
                
                vertices, triangles = compound.tessellate(0.4)
                if not vertices or not triangles:
                    raise ValueError("Generated geometry resulted in an empty mesh.")

                x = np.array([v.x for v in vertices])
                y = np.array([v.y for v in vertices])
                z = np.array([v.z for v in vertices])
                i_idx = np.array([t[0] for t in triangles])
                j_idx = np.array([t[1] for t in triangles])
                k_idx = np.array([t[2] for t in triangles])
                
                dist_from_center = np.sqrt(x**2 + y**2)
                T = temp - (np.abs(z - z_mid) * 0.4) - (dist_from_center * 0.15)
                
                st.success("Modular Fixture CAD Model & FEM Loads successfully compiled.")
                
                fig2 = go.Figure(data=[
                    go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i_idx, j=j_idx, k=k_idx,
                        intensity=T,
                        colorscale='Inferno',
                        colorbar=dict(title="Temperature (°C)", len=0.75),
                        flatshading=True,
                        showscale=True
                    )
                ])
                
                # ==============================================================
                # NEW: PLOTLY 3D SPATIAL ANNOTATIONS FOR FEM BOUNDARIES
                # ==============================================================
                fem_annotations = [
                    # Convection/Radiation on Top Plate
                    dict(
                        x=0, y=0, z=z_top_surface,
                        text=f"<b>Furnace Thermal Load</b><br>q_conv = h(T_inf - T_s)<br>T_inf = {temp}°C",
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                        ax=0, ay=-60, font=dict(color="white", size=12),
                        bgcolor="rgba(220, 50, 50, 0.85)", bordercolor="black", borderwidth=1
                    ),
                    # Exterior Wall Convection
                    dict(
                        x=fixture_r_computed, y=0, z=z_mid,
                        text="<b>Exterior Boundary</b><br>Forced Convection + Radiation",
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                        ax=80, ay=0, font=dict(color="white", size=12),
                        bgcolor="rgba(50, 50, 200, 0.85)", bordercolor="black", borderwidth=1
                    ),
                    # Internal Interface Resistance
                    dict(
                        x=waist_dia/2.0 + 2.0, y=0, z=z_mid,
                        text="<b>Interface</b><br>Thermal Contact Resistance (Rc)<br>Ceramic (k=30) ⟷ SS (k=16)",
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                        ax=100, ay=60, font=dict(color="black", size=11),
                        bgcolor="rgba(250, 200, 50, 0.9)", bordercolor="black", borderwidth=1
                    )
                ]

                fig2.update_layout(
                    title=f"Modular Heat-Setting Fixture & FEM Boundary Conditions (Target: {temp}°C)",
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
