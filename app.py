import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go
import cadquery as cq

st.set_page_config(page_title="Comprehensive Medical Device & Fixture Suite", layout="wide")

st.title("🔬 Integrated Dual-Layer Mechanics & Precision PDA Fixture Studio")
st.markdown("Complete computational and CAD engineering suite combining Nitinol braid kinematics, force optimization, and open-assembly heat-setting fixture generation.")

# Create Multi-Tab Layout for Complete Workflow
tab1, tab2 = st.tabs(["1. Dual-Layer Mechanics & Optimization Engine", "2. Open-Core CAD Fixture & Thermal Simulation"])

# ==============================================================================
# TAB 1: DUAL-LAYER MECHANICS, FORCE SIMULATION & OPTIMIZATION
# ==============================================================================
with tab1:
    st.header("Dual-Layer Occluder Structural & Force Modeller")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Macro Geometry & Inner Core")
        D_macro = st.slider("Waist Diameter (mm)", 4.0, 30.0, 12.0, 0.5, key="t1_D")
        P_macro = st.slider("Pitch Length / Pic Length (mm)", 1.0, 10.0, 4.0, 0.2, key="t1_P")
        N1 = st.slider("Inner Wire Count (N1)", 16, 72, 36, 4, key="t1_N1")
        d1 = st.slider("Inner Wire Diameter (mm)", 0.10, 0.30, 0.16, 0.01, key="t1_d1")

    with col_t2:
        st.subheader("Outer Shell & Displacement")
        N2 = st.slider("Outer Wire Count (N2)", 36, 144, 72, 4, key="t1_N2")
        d2 = st.slider("Outer Wire Diameter (mm)", 0.05, 0.15, 0.09, 0.01, key="t1_d2")
        delta_r = st.slider("Radial Displacement Simulation (mm)", 0.1, 5.0, 1.5, 0.1, key="t1_dr")

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

    st.subheader("Automated Design Space Optimization")
    if st.button("Run L-BFGS-B Optimization Routine", type="secondary"):
        with st.spinner("Optimizing wire parameters against target mechanical bounds..."):
            def objective(x):
                n_inner, d_inner, n_outer, d_outer = x
                k_i = 0.05 * (d_inner**4) * n_inner
                k_o = 0.03 * (d_outer**4) * n_outer
                f_val = (k_i * (1.5**1.2)) + (k_o * (1.5**1.1))
                ei_i = (n_inner * E_eff * np.pi * (d_inner**4) / 64.0)
                ei_o = (n_outer * E_eff * np.pi * (d_outer**4) / 64.0)
                ei_tot = ei_i + ei_o
                cost = ((f_val - 0.2) ** 2) * 150.0 + (ei_tot / 1200.0)
                return cost

            bounds_constraints = [(16, 72), (0.1, 0.25), (36, 144), (0.05, 0.15)]
            res = minimize(objective, [36, 0.16, 72, 0.09], bounds=bounds_constraints, method='L-BFGS-B')
            
            if res.success:
                opt_n1, opt_d1, opt_n2, opt_d2 = res.x
                st.success("Optimization Successfully Converged!")
                oc1, oc2 = st.columns(2)
                with oc1:
                    st.write(f"- **Optimized Inner Wire Count ($N_1$):** {int(round(opt_n1))}")
                    st.write(f"- **Optimized Inner Diameter ($d_1$):** {opt_d1:.3f} mm")
                with oc2:
                    st.write(f"- **Optimized Outer Wire Count ($N_2$):** {int(round(opt_n2))}")
                    st.write(f"- **Optimized Outer Diameter ($d_2$):** {opt_d2:.3f} mm")
            else:
                st.warning("Optimization did not converge within specified bounds.")


# ==============================================================================
# TAB 2: OPEN-CORE CAD FIXTURE & THERMAL SIMULATION
# ==============================================================================
with tab2:
    st.header("Precision PDA Occluder Core & Mandrel Fixture Studio")
    st.markdown("Generate open-assembly fixture CAD geometry, thermal distribution fields, and manufacturable STEP files.")

    cad_c1, cad_c2 = st.columns(2)
    with cad_c1:
        st.subheader("Device Parameters")
        pulm_dia = st.slider("Pulmonary End Diameter (D_pulm)", 4.0, 10.0, 6.0, step=0.5, key="t2_dpul")
        waist_dia = st.slider("Central Waist Diameter (D_waist)", 2.0, 8.0, 4.0, step=0.5, key="t2_dwst")
        aortic_dia = st.slider("Aortic Retention Disc Diameter (D_aortic)", 12.0, 30.0, 16.0, step=0.5, key="t2_daor")
        length = st.slider("Total Device Length (L)", 8.0, 25.0, 14.0, step=0.5, key="t2_len")
        wire_count = st.slider("Nitinol Wire Ends", 36, 144, 72, step=12, key="t2_wires")

    with cad_c2:
        st.subheader("Thermal Processing Setup")
        temp = st.number_input("Target Setting Temperature (°C)", 400, 600, 500, key="t2_temp")
        soak_time = st.number_input("Soak Time (mins)", 5, 60, 15, key="t2_soak")
        st.info("Configured for 17-4PH Stainless Steel or Graphite Open-Assembly Clamshell Fixtures.")

    def generate_pda_core_fixture(d_pulm, d_waist, d_aortic, lg, wires):
        center_bore_r = 3.0
        pulm_r = max(d_pulm / 2, center_bore_r + 1.0)
        waist_r = max(d_waist / 2, center_bore_r + 0.8)
        aortic_r = max(d_aortic / 2, waist_r + 2.0)
        
        plate_r = aortic_r + 12.0
        base_r = plate_r + 10.0
        
        assy = cq.Assembly(name="PDA_Core_Fixture_Assembly")

        # 1. Base Stand
        base = (cq.Workplane("XY")
                .circle(base_r).extrude(12)
                .faces(">Z").chamfer(1.5)
                .faces(">Z").workplane()
                .circle(plate_r).extrude(6)
                .faces(">Z").workplane().circle(center_bore_r).cutThruAll())
        assy.add(base, name="BaseStand", color=cq.Color(0.55, 0.55, 0.58))

        # 2. Bottom Retention Plate
        bottom_plate = (cq.Workplane("XY").circle(plate_r).extrude(4)
                        .faces(">Z").workplane().circle(center_bore_r).cutThruAll())
        bottom_plate = bottom_plate.faces(">Z").workplane().polarArray(plate_r - 4, 0, 360, 4).circle(1.6).cutThruAll()
        bottom_plate = bottom_plate.faces(">Z").workplane().polarArray(plate_r - 4, 45, 360, 2).circle(2.0).cutThruAll()
        
        for r_idx in range(2):
            curr_r = center_bore_r + 2.0 + (r_idx * ((pulm_r - center_bore_r - 2) / max(1, 2)))
            bottom_plate = bottom_plate.faces(">Z").workplane().polarArray(curr_r, 0, 360, 24).circle(0.35).cutThruAll()

        assy.add(bottom_plate, name="BottomRetentionPlate", loc=cq.Location(cq.Vector(0, 0, 18)), color=cq.Color(0.8, 0.8, 0.85))

        # 3. Top Retention Plate
        top_plate = (cq.Workplane("XY").circle(plate_r).extrude(4)
                     .faces(">Z").workplane().circle(center_bore_r).cutThruAll())
        top_plate = top_plate.faces(">Z").workplane().polarArray(plate_r - 4, 0, 360, 4).circle(1.6).cutThruAll()
        top_plate = top_plate.faces(">Z").workplane().polarArray(plate_r - 4, 45, 360, 2).circle(2.0).cutThruAll()
        
        for r_idx in range(3):
            curr_r = center_bore_r + 2.0 + (r_idx * ((aortic_r - center_bore_r - 2) / max(1, 3)))
            top_plate = top_plate.faces(">Z").workplane().polarArray(curr_r, 0, 360, 36).circle(0.35).cutThruAll()

        assy.add(top_plate, name="TopRetentionPlate", loc=cq.Location(cq.Vector(0, 0, 18 + 4 + lg)), color=cq.Color(0.8, 0.8, 0.85))

        # 4. Central Solid PDA Shaping Core (Mandrel)
        core_profile = (cq.Workplane("XZ")
                        .moveTo(center_bore_r, 0)
                        .lineTo(pulm_r, 0)
                        .threePointArc((waist_r, lg * 0.4), (aortic_r, lg))
                        .lineTo(center_bore_r, lg)
                        .close().revolve(360, (0, 0, 0), (0, 0, 1)))

        assy.add(core_profile, name="PDAShapingCore", loc=cq.Location(cq.Vector(0, 0, 22)), color=cq.Color(0.9, 0.6, 0.1))

        # 5. Fasteners and Tie Rods
        stud = cq.Workplane("XY").circle(1.5).extrude(lg + 28)
        nut = cq.Workplane("XY").polygon(6, 4.5).extrude(3).faces(">Z").workplane().circle(1.5).cutThruAll()
        pin = cq.Workplane("XY").circle(1.9).extrude(lg + 16)

        for i in range(4):
            angle = np.radians(i * 90)
            x, y = (plate_r - 4) * np.cos(angle), (plate_r - 4) * np.sin(angle)
            assy.add(stud, name=f"Bolt_{i}", loc=cq.Location(cq.Vector(x, y, 10)), color=cq.Color(0.7, 0.7, 0.7))
            assy.add(nut, name=f"BottomNut_{i}", loc=cq.Location(cq.Vector(x, y, 18)), color=cq.Color(0.75, 0.75, 0.75))
            assy.add(nut, name=f"TopNut_{i}", loc=cq.Location(cq.Vector(x, y, 18 + 4 + lg + 4)), color=cq.Color(0.75, 0.75, 0.75))

        for i in range(2):
            angle = np.radians(i * 180 + 45)
            x, y = (plate_r - 4) * np.cos(angle), (plate_r - 4) * np.sin(angle)
            assy.add(pin, name=f"AlignPin_{i}", loc=cq.Location(cq.Vector(x, y, 18)), color=cq.Color(0.4, 0.4, 0.4))

        return assy

    if st.button("🚀 Generate Open-Core CAD Fixture & Render Thermal Profile", type="primary", use_container_width=True):
        with st.spinner("Computing open-assembly CAD geometry and mapping thermal gradients..."):
            try:
                assy = generate_pda_core_fixture(pulm_dia, waist_dia, aortic_dia, length, wire_count)
                compound = assy.toCompound()
                
                vertices, triangles = compound.tessellate(0.2)
                
                if not vertices or not triangles:
                    raise ValueError("Generated geometry resulted in an empty mesh.")

                x = np.array([v.x for v in vertices])
                y = np.array([v.y for v in vertices])
                z = np.array([v.z for v in vertices])
                i_idx = np.array([t[0] for t in triangles])
                j_idx = np.array([t[1] for t in triangles])
                k_idx = np.array([t[2] for t in triangles])
                
                z_mid = 22 + (length / 2)
                dist_from_center = np.sqrt(x**2 + y**2)
                T = temp - (np.abs(z - z_mid) * 0.4) - (dist_from_center * 0.15)
                
                st.success("Open-Core PDA Fixture & Thermal Field Computed Successfully.")
                
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
                
                fig2.update_layout(
                    title=f"Open-Assembly PDA Shaping Core Fixture (Target: {temp}°C)",
                    scene=dict(
                        xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                        aspectmode="data"
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    height=700
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                filename = "PDA_Core_Mandrel_Fixture.step"
                cq.exporters.export(compound, filename)
                
                with open(filename, "rb") as file:
                    st.download_button(
                        label="💾 Download Fixture Manufacturing CAD (.STEP)",
                        data=file,
                        file_name=filename,
                        mime="application/step",
                        type="primary",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Engine Exception: {str(e)}")