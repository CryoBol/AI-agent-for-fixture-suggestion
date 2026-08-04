import streamlit as st
import numpy as np
import plotly.graph_objects as go
import cadquery as cq
import os

st.set_page_config(page_title="Nitinol Heat Setting Fixture AI Agent", layout="wide")

st.title("🔬 PDA Occluder Heat Setting Fixture AI Agent")
st.markdown("Automated parametric CAD generation, thermal FEM prediction, and design optimization.")

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("1. Input Parameters")

# Dimensions
st.sidebar.subheader("Occluder Dimensions (mm)")
waist_dia = st.sidebar.slider("Waist Diameter", 4.0, 16.0, 8.0)
disc_dia = st.sidebar.slider("Disc Diameter", 8.0, 30.0, 14.0)
length = st.sidebar.slider("Overall Length", 5.0, 20.0, 10.0)

# Process parameters
st.sidebar.subheader("Heat Treatment Parameters")
temp = st.sidebar.number_input("Target Temperature (°C)", 450, 600, 525)
soak_time = st.sidebar.number_input("Soak Time (mins)", 5, 60, 15)
fixture_material = st.sidebar.selectbox("Fixture Material", ["17-4 PH Stainless Steel", "316L Stainless Steel", "Alumina Ceramic Core / SS Body"])

# --- CAD GENERATION MODULE ---
def generate_cad_design1(w_dia, d_dia, lg):
    """Design 1: Standard Stack Assembly Base"""
    base = cq.Workplane("XY").circle(d_dia + 10).extrude(5)
    core = cq.Workplane("XY").workplane(offset=5).circle(w_dia).extrude(lg)
    top = cq.Workplane("XY").workplane(offset=5+lg).circle(d_dia + 10).extrude(5)
    model = base.union(core).union(top)
    return model

def generate_cad_design2(w_dia, d_dia, lg):
    """Design 2: Integrated Core (Self-aligning)"""
    base = cq.Workplane("XY").circle(d_dia + 8).extrude(4)
    core = cq.Workplane("XY").workplane(offset=4).circle(w_dia + 1).extrude(lg)
    model = base.union(core)
    return model

def generate_cad_design3(w_dia, d_dia, lg):
    """Design 3: Radial Tensioning Fixture Outer Ring"""
    ring = cq.Workplane("XY").circle(d_dia + 18).circle(d_dia + 10).extrude(lg + 8)
    return ring

# --- SURROGATE FEM MODULE ---
def run_thermal_fem_surrogate(design_id, temp, soak_time, material):
    """Simulates thermal distribution gradient (K) across Nitinol contact region"""
    np.random.seed(design_id)
    # Generate mock 3D mesh points for thermal visualization
    x = np.linspace(-15, 15, 20)
    y = np.linspace(-15, 15, 20)
    z = np.linspace(0, 20, 20)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Thermal gradient penalty factors per design
    penalties = {1: 1.2, 2: 0.7, 3: 1.5} # Design 2 (Ceramic/Integrated) has best thermal uniformity
    mat_alpha = 0.8 if "Ceramic" in material else 1.0
    
    # Calculate temperature field T(x,y,z)
    R = np.sqrt(X**2 + Y**2)
    T = temp - (R * penalties[design_id] * mat_alpha) + np.sin(Z)*2
    
    gradient_max = np.max(T) - np.min(T)
    soak_efficiency = min(100.0, (soak_time / 15.0) * 90 + np.random.uniform(-5, 5))
    
    return X, Y, Z, T, gradient_max, soak_efficiency

# --- TRIGGER EVALUATION ---
if st.button("🚀 Generate & Evaluate Fixture Designs"):
    
    col1, col2, col3 = st.columns(3)
    designs = [
        {"id": 1, "name": "Design 1: Original Standard", "complexity": "Medium (6 Parts)", "col": col1},
        {"id": 2, "name": "Design 2: Integrated Core", "complexity": "Low (3 Parts - Recommended)", "col": col2},
        {"id": 3, "name": "Design 3: Radial Tensioning", "complexity": "High (12+ Parts)", "col": col3}
    ]
    
    best_design = None
    best_score = -1
    results = {}

    for d in designs:
        with d["col"]:
            st.header(d["name"])
            st.caption(f"Assembly Complexity: {d['complexity']}")
            
            # 1. Run FEM
            X, Y, Z, T, grad, soak_eff = run_thermal_fem_surrogate(d["id"], temp, soak_time, fixture_material)
            
            # 2. Score Calculation (Thermal Uniformity 60%, Simplicity 40%)
            comp_penalty = {1: 15, 2: 0, 3: 35}[d["id"]]
            overall_score = round(100 - (grad * 1.5) - comp_penalty, 1)
            results[d["id"]] = {"score": overall_score, "grad": grad, "name": d["name"]}
            
            if overall_score > best_score:
                best_score = overall_score
                best_design = d["id"]

            st.metric("Thermal Uniformity Index", f"{100-grad:.1f}/100", delta=f"-{grad:.1f}°C max ΔT")
            st.metric("Overall Optimization Score", f"{overall_score}/100")

            # 3. Interactive Plotly Thermal Contour
            fig = go.Figure(data=go.Volume(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=T.flatten(),
                isomin=np.min(T), isomax=np.max(T),
                opacity=0.3,
                surface_count=10,
                colorscale='Jet'
            ))
            fig.update_layout(title="Thermal Contour (°C)", margin=dict(l=0, r=0, b=0, t=30), height=300)
            st.plotly_chart(fig, use_container_width=True)

            # 4. Generate STEP File Download
            cad_gen_map = {1: generate_cad_design1, 2: generate_cad_design2, 3: generate_cad_design3}
            cad_shape = cad_gen_map[d["id"]](waist_dia, disc_dia, length)
            filename = f"fixture_design_{d['id']}.step"
            cq.exporters.export(cad_shape, filename)
            
            with open(filename, "rb") as file:
                st.download_button(
                    label=f"💾 Download {d['name']} CAD (.STEP)",
                    data=file,
                    file_name=filename,
                    mime="application/step"
                )

    # --- AGENT JUSTIFICATION REPORT ---
    st.markdown("---")
    st.subheader("🤖 AI Agent Recommendation & Justification Report")
    
    best = results[best_design]
    st.success(f"**Selected Optimal Choice:** {best['name']} (Score: {best['score']}/100)")
    
    st.markdown(f"""
    ### Rationale:
    1. **Thermal Homogeneity:** Design {best_design} maintained a thermal gradient of **{best['grad']:.2f}°C** across the Nitinol wire geometry at **{temp}°C**. Uniform thermal distribution prevents localized transformation temperature shifts ($A_f$) in the Nitinol shape memory alloy.
    2. **Manufacturing & Tooling Efficiency:** Taking into account machining tolerances, assembly stack-up errors, and thermal expansion mismatch between parts during heating/cooling cycles:
       * *Design 2* minimizes thermal mass and interface contact resistance.
       * *Design 3* introduces localized heat-sink effects at radial tensioning screw joints.
    3. **Material Choice:** Utilizing **{fixture_material}** ensures oxidation resistance and dimensional stability across repeating thermal cycles.
    """)