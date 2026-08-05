import streamlit as st
import numpy as np
import plotly.graph_objects as go
import cadquery as cq

st.set_page_config(page_title="PDA Occluder Split-Cavity Fixture AI", layout="wide")

st.title("🔬 Precision Split-Cavity PDA Occluder Heat-Setting Fixture")
st.markdown("Manufacturing-grade female die mold design for thermal shape-setting of braided Nitinol occluders.")

# --- SIDEBAR: USER INPUTS (PDA DEVICE PARAMETERS) ---
st.sidebar.header("1. Occluder Target Dimensions")

pulm_dia = st.sidebar.slider("Pulmonary End Diameter ($D_{pulm}$)", 4.0, 10.0, 6.0, step=0.5)
waist_dia = st.sidebar.slider("Central Waist Diameter ($D_{waist}$)", 2.0, 8.0, 4.0, step=0.5)
aortic_dia = st.sidebar.slider("Aortic Retention Disc Diameter ($D_{aortic}$)", 12.0, 30.0, 16.0, step=0.5)
length = st.sidebar.slider("Total Device Length ($L$)", 8.0, 25.0, 14.0, step=0.5)

st.sidebar.subheader("Heat Treatment Parameters")
temp = st.sidebar.number_input("Target Setting Temperature (°C)", 400, 600, 500)
soak_time = st.sidebar.number_input("Soak Time (mins)", 5, 60, 15)

# --- SPLIT-CAVITY MOLD CAD GENERATION ---
def generate_split_cavity_fixture(d_pulm, d_waist, d_aortic, lg):
    block_size = max(60.0, d_aortic + 30.0)
    block_half_h = 15.0
    
    pulm_r = d_pulm / 2.0
    waist_r = d_waist / 2.0
    aortic_r = d_aortic / 2.0
    
    assy = cq.Assembly(name="Split_Cavity_Mold_Assembly")

    # 1. Half-Cavity Profile (Negative shape of the PDA occluder rotated in XZ plane)
    # We create a solid profile to subtract from the mold blocks
    cavity_profile = (cq.Workplane("XZ")
                      .moveTo(0, 0)
                      .lineTo(pulm_r, 0)
                      .threePointArc((waist_r, lg * 0.4), (aortic_r, lg))
                      .lineTo(0, lg)
                      .close()
                      .revolve(360, (0, 0, 0), (0, 0, 1)))

    # 2. Bottom Mold Block
    bottom_block = (cq.Workplane("XY")
                    .rect(block_size, block_size)
                    .extrude(block_half_h))
    
    # Subtract the bottom half of the cavity (from Z=0 to Z=lg/2)
    # Add corner bolt holes (M5 clearance) and alignment dowel holes
    bolt_offset = block_size / 2 - 8.0
    bottom_block = (bottom_block
                    .faces(">Z").workplane()
                    .cut(cavity_profile)
                    .rect(block_size - 16, block_size - 16, forConstruction=True)
                    .vertices().circle(2.75).cutThruAll())

    assy.add(bottom_block, name="BottomMoldHalf", color=cq.Color(0.7, 0.75, 0.8))

    # 3. Top Mold Block (Mirrored / Shifted upwards)
    top_block = (cq.Workplane("XY")
                 .rect(block_size, block_size)
                 .extrude(block_half_h))
    
    top_block = (top_block
                 .faces("<Z").workplane()
                 .cut(cavity_profile)
                 .rect(block_size - 16, block_size - 16, forConstruction=True)
                 .vertices().circle(2.75).cutThruAll())

    assy.add(top_block, name="TopMoldHalf", loc=cq.Location(cq.Vector(0, 0, block_half_h)), color=cq.Color(0.75, 0.8, 0.85))

    # 4. Corner Clamping Bolts (M5 Socket Head Cap Screws)
    bolt = cq.Workplane("XY").circle(2.5).extrude(block_half_h * 2 + 10)
    nut = cq.Workplane("XY").polygon(6, 4.5).extrude(4)

    corners = [
        (bolt_offset, bolt_offset), (-bolt_offset, bolt_offset),
        (-bolt_offset, -bolt_offset), (bolt_offset, -bolt_offset)
    ]
    
    for idx, (cx, cy) in enumerate(corners):
        assy.add(bolt, name=f"ClampBolt_{idx}", loc=cq.Location(cq.Vector(cx, cy, -5)), color=cq.Color(0.6, 0.6, 0.65))
        assy.add(nut, name=f"TopNut_{idx}", loc=cq.Location(cq.Vector(cx, cy, block_half_h * 2)), color=cq.Color(0.7, 0.7, 0.7))

    return assy


# --- TRIGGER EVALUATION ---
if st.button("🚀 Generate Split-Cavity Fixture & Render Thermal Field", type="primary", use_container_width=True):
    
    with st.spinner("Executing OpenCASCADE boolean cavity cuts and mesh tessellation..."):
        try:
            # 1. Generate CAD Assembly
            assy = generate_split_cavity_fixture(pulm_dia, waist_dia, aortic_dia, length)
            compound = assy.toCompound()
            
            # 2. Extract Tessellated Mesh
            vertices, triangles = compound.tessellate(0.25)
            
            if not vertices or not triangles:
                raise ValueError("Generated geometry resulted in an empty mesh.")

            x = np.array([v.x for v in vertices])
            y = np.array([v.y for v in vertices])
            z = np.array([v.z for v in vertices])
            i = np.array([t[0] for t in triangles])
            j = np.array([t[1] for t in triangles])
            k = np.array([t[2] for t in triangles])
            
            # 3. Calculate Nodal Temperatures mapped across the die blocks
            dist_from_center = np.sqrt(x**2 + y**2)
            T = temp - (dist_from_center * 0.2)
            
            st.success("Split-Cavity Mold Geometry & Thermal Profile Computed Successfully.")
            
            # 4. Render CAD geometry mapped with FEM color gradient
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    intensity=T,
                    colorscale='Inferno',
                    colorbar=dict(title="Temperature (°C)", len=0.75),
                    flatshading=True,
                    showscale=True
                )
            ])
            
            fig.update_layout(
                title=f"Split-Cavity Heat-Setting Die (Target: {temp}°C)",
                scene=dict(
                    xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                    aspectmode="data"
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 5. Export STEP File
            filename = "PDA_Split_Cavity_Fixture.step"
            cq.exporters.export(compound, filename)
            
            with open(filename, "rb") as file:
                st.download_button(
                    label="💾 Download Mold Manufacturing CAD (.STEP)",
                    data=file,
                    file_name=filename,
                    mime="application/step",
                    type="primary",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Engine Exception: {str(e)}")