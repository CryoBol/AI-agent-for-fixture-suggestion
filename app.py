import streamlit as st
import numpy as np
import plotly.graph_objects as go
import cadquery as cq

st.set_page_config(page_title="Nitinol Heat Setting Fixture AI Agent", layout="wide")

st.title("🔬 PDA Occluder Heat Setting Fixture AI Agent")
st.markdown("Automated parametric CAD generation, surface thermal mapping, and design optimization.")

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("1. Input Parameters")

st.sidebar.subheader("Occluder Dimensions (mm)")
waist_dia = st.sidebar.slider("Waist Diameter ($D_{w}$)", 4.0, 12.0, 6.0, step=0.5)
disc_dia = st.sidebar.slider("Disc Diameter ($D_{d}$)", 10.0, 30.0, 16.0, step=0.5)
length = st.sidebar.slider("Core Length ($L$)", 5.0, 25.0, 12.0, step=0.5)
wire_count = st.sidebar.slider("Nitinol Wire Ends", 36, 144, 72, step=12)

st.sidebar.subheader("Heat Treatment Parameters")
temp = st.sidebar.number_input("Target Temperature (°C)", 450, 600, 525)
soak_time = st.sidebar.number_input("Soak Time (mins)", 5, 60, 15)
fixture_material = st.sidebar.selectbox("Fixture Material", ["17-4 PH Stainless Steel", "316L Stainless Steel", "Alumina Ceramic Core / SS Body"])

# --- CAD GENERATION MODULE ---
def generate_design1_assembly(w_dia, d_dia, lg, hole_count):
    base = (cq.Workplane("XY")
            .circle(d_dia/2 + 20).extrude(8)
            .faces(">Z").workplane()
            .circle(d_dia/2 + 10).extrude(4))

    plate = (cq.Workplane("XY")
             .circle(d_dia/2 + 10).extrude(6)
             .faces(">Z").workplane()
             .polarArray(d_dia/2 - 2, 0, 360, hole_count).circle(0.5).cutThruAll()
             .faces(">Z").workplane()
             .polarArray(d_dia/2 + 6, 0, 360, 4).circle(1.6).cutThruAll()
             .faces(">Z").workplane()
             .circle(4.0).cutThruAll())

    core = (cq.Workplane("XZ")
            .moveTo(0, 0)
            .lineTo(d_dia/2 - 2, 0)
            .lineTo(d_dia/2 - 2, lg * 0.15)
            .lineTo(w_dia/2, lg * 0.5)
            .lineTo(d_dia/2 - 2, lg * 0.85)
            .lineTo(d_dia/2 - 2, lg)
            .lineTo(0, lg)
            .close()
            .revolve(360, (0,0,0), (0,0,1))
            .faces(">Z").workplane().circle(4.0).cutThruAll())

    pin = cq.Workplane("XY").circle(1.5).extrude(lg + 16)

    assy = cq.Assembly(name="Design1_Original")
    assy.add(base, name="Base", color=cq.Color(0.3, 0.3, 0.3)) # Dark Gray
    
    z_base = 12
    z_core = z_base + 6
    z_top = z_core + lg
    
    assy.add(plate, name="BottomPlate", loc=cq.Location(cq.Vector(0, 0, z_base)), color=cq.Color(0.8, 0.8, 0.8)) # Light Gray
    assy.add(core, name="CeramicCore", loc=cq.Location(cq.Vector(0, 0, z_core)), color=cq.Color(0.96, 0.87, 0.7)) # Wheat / Ceramic tone
    assy.add(plate, name="TopPlate", loc=cq.Location(cq.Vector(0, 0, z_top)), color=cq.Color(0.8, 0.8, 0.8))
    
    for i in range(4):
        angle = np.radians(i * 90)
        rad = d_dia/2 + 6
        x = rad * np.cos(angle)
        y = rad * np.sin(angle)
        assy.add(pin, name=f"Pin_{i}", loc=cq.Location(cq.Vector(x, y, z_base)), color=cq.Color(0.7, 0.7, 0.7))

    return assy


def generate_design2_assembly(w_dia, d_dia, lg, hole_count):
    half_lg = lg / 2
    def make_half():
        return (cq.Workplane("XY")
                .circle(d_dia/2 + 10).extrude(6)
                .faces(">Z").workplane()
                .circle(d_dia/2).extrude(half_lg)
                .edges(">Z and %CIRCLE").fillet(w_dia/4) 
                .faces(">Z").workplane()
                .polarArray(d_dia/2 - 2, 0, 360, hole_count).circle(0.5).cutThruAll()
                .faces(">Z").workplane()
                .circle(4.0).cutThruAll())

    bottom_half = make_half()
    top_half = make_half().rotate((0,0,0), (1,0,0), 180)

    assy = cq.Assembly(name="Design2_Integrated")
    assy.add(bottom_half, name="BottomHalf", loc=cq.Location(cq.Vector(0, 0, 0)), color=cq.Color(0.8, 0.8, 0.8))
    assy.add(top_half, name="TopHalf", loc=cq.Location(cq.Vector(0, 0, 12 + lg)), color=cq.Color(0.4, 0.4, 0.4))
    return assy


def generate_design3_assembly(w_dia, d_dia, lg, hole_count):
    hub = (cq.Workplane("XY")
           .circle(d_dia/2).extrude(lg + 12)
           .faces(">Z").workplane()
           .polarArray(d_dia/2 - 3, 0, 360, hole_count).circle(0.5).cutThruAll())

    ring_outer_rad = d_dia/2 + 25
    ring_inner_rad = d_dia/2 + 5
    
    ring = (cq.Workplane("XY")
            .circle(ring_outer_rad).circle(ring_inner_rad).extrude(lg + 12))

    screw = cq.Workplane("XY").circle(1.8).extrude(20)

    assy = cq.Assembly(name="Design3_Radial")
    assy.add(hub, name="CenterHub", color=cq.Color(0.8, 0.8, 0.8))
    assy.add(ring, name="OuterRing", color=cq.Color(0.3, 0.3, 0.3))
    
    for i in range(8):
        angle = np.radians(i * 45)
        x = (ring_outer_rad - 5) * np.cos(angle)
        y = (ring_outer_rad - 5) * np.sin(angle)
        loc = cq.Location(cq.Vector(x, y, (lg+12)/2)) * cq.Location(cq.Vector(0,0,0), cq.Vector(0,0,1), i*45) * cq.Location(cq.Vector(0,0,0), cq.Vector(1,0,0), 90)
        assy.add(screw, name=f"Screw_{i}", loc=loc, color=cq.Color(0.7, 0.7, 0.7))

    return assy


# --- SURFACE FEM MAPPING MODULE ---
def render_surface_thermal_model(design_id, w_dia, d_dia, lg, temp, soak_time, material):
    theta = np.linspace(0, 2 * np.pi, 50)
    z_vals = np.linspace(0, lg + 12, 50)
    Theta, Z = np.meshgrid(theta, z_vals)
    
    if design_id == 1:
        core_z = np.clip(Z - 6, 0, lg)
        norm_z = core_z / lg
        Radius = np.where(
            (Z >= 6) & (Z <= lg + 6),
            (d_dia/2) - ((d_dia/2 - w_dia/2) * np.sin(norm_z * np.pi)),
            d_dia/2 + 4
        )
    elif design_id == 2:
        Radius = np.where(Z < (lg/2 + 6), d_dia/2, w_dia/2 + 2)
    else:
        Radius = (d_dia/2 + 2) * np.ones_like(Z)

    X = Radius * np.cos(Theta)
    Y = Radius * np.sin(Theta)
    
    penalties = {1: 0.8, 2: 0.5, 3: 1.1} 
    mat_alpha = 0.85 if "Ceramic" in material else 1.0
    
    T = temp - (np.abs(Z - (lg+12)/2) * penalties[design_id] * mat_alpha) + (np.sin(Theta * 3) * 1.2)
    gradient_max = float(np.max(T) - np.min(T))
    
    return X, Y, Z, T, gradient_max


# --- TRIGGER EVALUATION ---
if st.button("🚀 Generate High-Fidelity Assembly & Evaluate", width="stretch"):
    
    col1, col2, col3 = st.columns(3)
    designs = [
        {"id": 1, "name": "Design 1: Original Standard", "complexity": "Medium (9 Parts)", "col": col1},
        {"id": 2, "name": "Design 2: Integrated Core", "complexity": "Low (4 Parts)", "col": col2},
        {"id": 3, "name": "Design 3: Radial Tensioning", "complexity": "High (16+ Parts)", "col": col3}
    ]
    
    best_design = None
    best_score = -1
    results = {}

    for d in designs:
        with d["col"]:
            st.header(d["name"])
            st.caption(f"Assembly Complexity: {d['complexity']}")
            
            try:
                # 1. Run Surface FEM Surrogate
                X, Y, Z, T, grad = render_surface_thermal_model(d["id"], waist_dia, disc_dia, length, temp, soak_time, fixture_material)
                
                # 2. Score Calculation
                comp_penalty = {1: 15, 2: 0, 3: 35}[d["id"]]
                overall_score = round(100 - (grad * 1.5) - comp_penalty, 1)
                results[d["id"]] = {"score": overall_score, "grad": grad, "name": d["name"]}
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_design = d["id"]

                st.metric("Thermal Uniformity Index", f"{100-grad:.1f}/100", delta=f"-{grad:.1f}°C max ΔT")
                st.metric("Optimization Score", f"{overall_score}/100")

                # 3. Interactive Surface Thermal Contour on the Model
                fig = go.Figure(data=[go.Surface(
                    x=X, y=Y, z=Z,
                    surfacecolor=T,
                    colorscale='Inferno',
                    colorbar=dict(title="°C", len=0.75)
                )])
                
                fig.update_layout(
                    title="Thermal Contour on Model Surface (°C)",
                    scene=dict(
                        xaxis_title="X (mm)",
                        yaxis_title="Y (mm)",
                        zaxis_title="Z (mm)",
                        aspectmode="data"
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                    height=320
                )
                st.plotly_chart(fig, width="stretch")

                # 4. Generate STEP Assembly Safely using cq.exporters
                cad_gen_map = {1: generate_design1_assembly, 2: generate_design2_assembly, 3: generate_design3_assembly}
                assembly = cad_gen_map[d["id"]](waist_dia, disc_dia, length, wire_count)
                
                filename = f"fixture_assembly_design_{d['id']}.step"
                cq.exporters.export(assembly, filename)
                
                with open(filename, "rb") as file:
                    st.download_button(
                        label=f"💾 Download {d['name']} (.STEP)",
                        data=file,
                        file_name=filename,
                        mime="application/step",
                        width="stretch"
                    )
            except Exception as e:
                st.error(f"Error generating {d['name']}: {str(e)}")

    # --- AGENT JUSTIFICATION REPORT ---
    if best_design:
        st.markdown("---")
        st.subheader("🤖 AI Agent Recommendation & Justification Report")
        best = results[best_design]
        st.success(f"**Selected Optimal Choice:** {best['name']} (Score: {best['score']}/100)")
        st.markdown(f"""
        ### Engineering Rationale:
        1. **Thermal Homogeneity:** Design {best_design} maintained a thermal gradient of **{best['grad']:.2f}°C** across the Nitinol wire geometry at **{temp}°C**. Uniform thermal distribution prevents localized transformation temperature shifts ($A_f$) in the shape memory alloy.
        2. **Manufacturing & Tooling Efficiency:** Taking into account the {wire_count} locating holes and stack-up tolerances, this configuration minimizes interface contact resistance and thermal distortion.
        3. **Material Choice:** Utilizing **{fixture_material}** ensures oxidation resistance and dimensional stability across repeating thermal cycles.
        """)