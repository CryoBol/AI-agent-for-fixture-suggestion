import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go
import cadquery as cq

st.set_page_config(page_title="PDA Occluder Heat-Setting Fixture Suite", layout="wide")

st.title("🔬 PDA Occluder Heat-Setting Fixture & Dual-Layer Mechanics Studio")
st.markdown("Integrated computational platform conforming to Drawing No. **PDHIF-01-ASSY** (Design 1: Original Standard Modular Fixture). Combines Nitinol braid mechanics, optimization, and robust parametric CAD fixture generation.")

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
        D_macro = st.slider("Waist Diameter (D_waist mm)", 4.0, 30.0, 12.0, 0.5, key="t1_D")
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
                    st.write(f"- **Optimized Inner Wire Count (N1):** {int(round(opt_n1))}")
                    st.write(f"- **Optimized Inner Diameter (d1):** {opt_d1:.3f} mm")
                with oc2:
                    st.write(f"- **Optimized Outer Wire Count (N2):** {int(round(opt_n2))}")
                    st.write(f"- **Optimized Outer Diameter (d2):** {opt_d2:.3f} mm")
            else:
                st.warning("Optimization did not converge within specified bounds.")


# ==============================================================================
# TAB 2: MODULAR HEAT-SETTING FIXTURE CAD & THERMAL STUDIO (PDHIF-01-ASSY)
# ==============================================================================
with tab2:
    st.header("Modular Heat-Setting Fixture Studio (Drawing No. PDHIF-01-ASSY)")
    st.markdown("Precision parametric CAD generation based on the 10-part modular architecture: Clamping Plates, Cavity Inserts, Replaceable Ceramic Waist Core, Spacer Ring, Dowel Pins, and M6 Shoulder Bolts.")

    cad_c1, cad_c2 = st.columns(2)
    with cad_c1:
        st.subheader("Device & Fixture Parameters")
        disc_dia = st.slider("Disc Outer Diameter (D_disc mm)", 12.0, 30.0, 26.0, step=1.0, key="t2_ddisc")
        waist_dia = st.slider("Waist Diameter (D_waist mm)", 4.0, 12.0, 8.0, step=1.0, key="t2_dwst")
        h_total = st.slider("Total Fixture Height (H_total mm)", 15.0, 30.0, 20.0, step=1.0, key="t2_htot")
        h_disc = st.slider("Disc Cavity Height (H_disc mm)", 4.0, 10.0, 6.0, step=0.5, key="t2_hdisc")
        h_waist = st.slider("Waist Core Height (H_waist mm)", 4.0, 10.0, 6.0, step=0.5, key="t2_hwaist")

    with cad_c2:
        st.subheader("Thermal Processing Setup")
        temp = st.number_input("Target Setting Temperature (°C)", 400, 600, 500, key="t2_temp")
        soak_time = st.number_input("Soak Time (mins)", 5, 60, 15, key="t2_soak")
        st.info("**Materials & Finish (Sec. 8):** Parts 1, 2, 5, 6, 7, 10: 17-4 PH SS (H900, Ra <= 0.4 µm). Part 4: Alumina Ceramic 99.7% (Ra <= 0.2 µm). Part 3: Dowels (6.0 h7).")

    def generate_modular_pda_fixture(d_disc, d_waist, h_tot, h_d, h_w):
        plate_r = (d_disc / 2.0) + 10.0
        bolt_circle_r = plate_r - 5.0
        bore_r = d_waist / 2.0
        disc_r = d_disc / 2.0

        assy = cq.Assembly(name="PDHIF_01_ASSY")

        # 1. Bottom Support Plate (Part 6) - 17-4 PH Stainless Steel
        bottom_plate = (cq.Workplane("XY")
                        .circle(plate_r).extrude(6.0)
                        .faces(">Z").workplane()
                        .circle(plate_r - 2.0).extrude(2.0))
        for i in range(4):
            ang = np.radians(i * 90)
            bx, by = bolt_circle_r * np.cos(ang), bolt_circle_r * np.sin(ang)
            bottom_plate = bottom_plate.faces(">Z").workplane().transformed(offset=cq.Vector(bx, by, 0)).circle(3.3).cutThruAll()
        for i in range(8):
            ang = np.radians(i * 45 + 22.5)
            vx, vy = (disc_r - 4) * np.cos(ang), (disc_r - 4) * np.sin(ang)
            bottom_plate = bottom_plate.faces(">Z").workplane().transformed(offset=cq.Vector(vx, vy, 0)).circle(1.0).cutThruAll()

        assy.add(bottom_plate, name="BottomSupportPlate", loc=cq.Location(cq.Vector(0, 0, 0)), color=cq.Color(0.75, 0.75, 0.8))

        # 2. Compression Stop / Spacer Ring (Part 10) - 17-4 PH Stainless Steel
        comp_stop = (cq.Workplane("XY")
                     .circle(disc_r + 2.0).extrude(1.0)
                     .faces(">Z").workplane().circle(disc_r - 1.0).cutThruAll())
        assy.add(comp_stop, name="CompressionStop", loc=cq.Location(cq.Vector(0, 0, 8.0)), color=cq.Color(0.7, 0.7, 0.75))

        # 3. Bottom Cavity Insert / Disc Former (Part 5) - Robust profile generation without selector errors
        bot_insert = (cq.Workplane("XZ")
                      .moveTo(bore_r + 0.05, 0)
                      .lineTo(disc_r + 0.1, 0)
                      .lineTo(disc_r + 0.1, h_d)
                      .lineTo(bore_r + 0.05, h_d)
                      .close()
                      .revolve(360, (0, 0, 0), (0, 0, 1)))
        assy.add(bot_insert, name="BottomCavityInsert", loc=cq.Location(cq.Vector(0, 0, 9.0)), color=cq.Color(0.85, 0.85, 0.9))

        # 4. Ceramic Waist Core (Part 4) - Alumina Ceramic 99.7%
        ceramic_core_solid = (cq.Workplane("XY").circle(bore_r + 1.2).extrude(h_w)
                              .faces(">Z").chamfer(0.5)
                              .faces("<Z").chamfer(0.5)
                              .faces(">Z").workplane().circle(bore_r).cutThruAll())
        assy.add(ceramic_core_solid, name="CeramicWaistCore", loc=cq.Location(cq.Vector(0, 0, 9.0 + h_d)), color=cq.Color(0.95, 0.95, 0.90))

        # 5. Top Cavity Insert / Disc Former (Part 2) - 17-4 PH Stainless Steel
        top_insert = (cq.Workplane("XZ")
                      .moveTo(bore_r + 0.05, 0)
                      .lineTo(disc_r + 0.1, 0)
                      .lineTo(disc_r + 0.1, h_d)
                      .lineTo(bore_r + 0.05, h_d)
                      .close()
                      .revolve(360, (0, 0, 0), (0, 0, 1)))
        assy.add(top_insert, name="TopCavityInsert", loc=cq.Location(cq.Vector(0, 0, 9.0 + h_d + h_w)), color=cq.Color(0.85, 0.85, 0.9))

        # 6. Top Clamping Plate (Part 1) - 17-4 PH Stainless Steel
        top_plate = (cq.Workplane("XY")
                     .circle(plate_r).extrude(6.0))
        for i in range(4):
            ang = np.radians(i * 90)
            bx, by = bolt_circle_r * np.cos(ang), bolt_circle_r * np.sin(ang)
            top_plate = top_plate.faces(">Z").workplane().transformed(offset=cq.Vector(bx, by, 0)).circle(3.3).cutThruAll()
        for i in range(12):
            ang = np.radians(i * 30)
            vx, vy = (disc_r - 6) * np.cos(ang), (disc_r - 6) * np.sin(ang)
            top_plate = top_plate.faces(">Z").workplane().transformed(offset=cq.Vector(vx, vy, 0)).circle(1.0).cutThruAll()
        top_plate = top_plate.faces(">Z").workplane().circle(disc_r - 2.0).cutThruAll()

        top_plate_z = 9.0 + h_d + h_w + h_d
        assy.add(top_plate, name="TopClampingPlate", loc=cq.Location(cq.Vector(0, 0, top_plate_z)), color=cq.Color(0.75, 0.75, 0.8))

        # 7. Shoulder Bolts (Part 7, 4 Nos. M6) & Dowel Pins (Part 3, 2 Nos.)
        bolt_len = top_plate_z + 6.0
        shoulder_bolt = cq.Workplane("XY").circle(3.0).extrude(bolt_len).faces(">Z").cylinder(3.0, 4.5)
        dowel_pin = cq.Workplane("XY").circle(3.0).extrude(top_plate_z)

        for i in range(4):
            ang = np.radians(i * 90)
            bx, by = bolt_circle_r * np.cos(ang), bolt_circle_r * np.sin(ang)
            assy.add(shoulder_bolt, name=f"ShoulderBolt_{i}", loc=cq.Location(cq.Vector(bx, by, 0)), color=cq.Color(0.6, 0.6, 0.65))

        for i in range(2):
            ang = np.radians(i * 180 + 45)
            dx, dy = (bolt_circle_r - 4.0) * np.cos(ang), (bolt_circle_r - 4.0) * np.sin(ang)
            assy.add(dowel_pin, name=f"DowelPin_{i}", loc=cq.Location(cq.Vector(dx, dy, 6.0)), color=cq.Color(0.4, 0.4, 0.45))

        return assy

    if st.button("🚀 Generate Modular Fixture CAD & Thermal Field", type="primary", use_container_width=True):
        with st.spinner("Compiling modular CAD assembly and mapping thermal diffusion field..."):
            try:
                assy = generate_modular_pda_fixture(disc_dia, waist_dia, h_total, h_disc, h_waist)
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
                
                z_mid = (9.0 + h_disc + (h_waist / 2.0))
                dist_from_center = np.sqrt(x**2 + y**2)
                T = temp - (np.abs(z - z_mid) * 0.35) - (dist_from_center * 0.1)
                
                st.success("Modular Fixture CAD Model & Thermal Distribution Successfully Computed.")
                
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
                    title=f"Modular Heat-Setting Fixture Assembly (Drawing No. PDHIF-01-ASSY | Target: {temp}°C)",
                    scene=dict(
                        xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                        aspectmode="data"
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    height=700
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
